"""Agent-native Fitness Coach Core tools backed by Agent Core DB.

Fitness Coach Core is the local canonical module for personal coaching in
single-tenant agents: user goals, nutrition, foods, routines, workouts, sets,
body metrics, check-ins, and progress summaries. External food databases,
wearables, and fitness apps are adapters/caches; structured Agent Core SQL
remains the source of truth for coaching decisions.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error

FITNESS_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: owner_id, "
    "source_channel, external_ref, labels, notes, source_confidence."
)


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


def _user() -> str:
    return sql.runtime_env().get("FITNESS_DB_RUNTIME_USER", "fitness_runtime")


def _check_fitness() -> bool:
    try:
        if not sql.enabled():
            return False
        sql.psql("SELECT 1;", user=_user())
        return True
    except Exception:
        return False


def _q(v: Any) -> str:
    return sql.quote_literal(v)


def _j(v: Any) -> str:
    return sql.quote_jsonb(v)


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}-{sql.slugify(value)}"


def _num(v: Any, default: str = "NULL") -> str:
    if v is None or v == "":
        return default
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {v!r}")


def _int(v: Any, default: str = "NULL") -> str:
    if v is None or v == "":
        return default
    try:
        return str(int(v))
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer value: {v!r}")


def _money(v: Any) -> float:
    return round(float(v or 0), 6)


def _estimated_1rm(weight_kg: Any, reps: Any) -> float | None:
    weight = float(weight_kg or 0)
    rep_count = int(reps or 0)
    if weight <= 0 or rep_count <= 0:
        return None
    if rep_count == 1:
        return _money(weight)
    return _money(weight * (1 + rep_count / 30.0))


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required or []}}}


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": FITNESS_METADATA_DESCRIPTION}}


def _open_food_facts_user_agent() -> str:
    env = sql.runtime_env()
    return env.get("FITNESS_OPEN_FOOD_FACTS_USER_AGENT") or os.getenv("FITNESS_OPEN_FOOD_FACTS_USER_AGENT") or "SitioUnoZeusFitnessCoach/0.1 (contact: support@sitiouno.com)"


def _off_product(barcode: str) -> dict[str, Any] | None:
    url = f"https://world.openfoodfacts.org/api/v2/product/{urllib.parse.quote(barcode)}.json"
    req = urllib.request.Request(url, headers={"User-Agent": _open_food_facts_user_agent()})
    with urllib.request.urlopen(req, timeout=12) as response:  # nosec B310 - fixed HTTPS API endpoint
        data = json.load(response)
    if data.get("status") != 1:
        return None
    product = data.get("product") or {}
    nutriments = product.get("nutriments") or {}
    return {
        "source_key": "open_food_facts",
        "external_id": barcode,
        "barcode": barcode,
        "name": product.get("product_name") or product.get("generic_name") or f"Open Food Facts {barcode}",
        "brand": product.get("brands"),
        "serving_size_g": _safe_float(product.get("serving_quantity")),
        "calories_per_100g": _safe_float(nutriments.get("energy-kcal_100g")),
        "protein_g_per_100g": _safe_float(nutriments.get("proteins_100g")),
        "carbs_g_per_100g": _safe_float(nutriments.get("carbohydrates_100g")),
        "fat_g_per_100g": _safe_float(nutriments.get("fat_100g")),
        "fiber_g_per_100g": _safe_float(nutriments.get("fiber_100g")),
        "sugar_g_per_100g": _safe_float(nutriments.get("sugars_100g")),
        "sodium_mg_per_100g": _safe_float(nutriments.get("sodium_100g"), multiplier=1000),
        "metadata": {"source": "open_food_facts", "license": "ODbL/DbCL; images CC BY-SA", "raw_product": product},
    }


def _safe_float(value: Any, multiplier: float = 1.0) -> float | None:
    try:
        if value in (None, ""):
            return None
        return _money(float(value) * multiplier)
    except (TypeError, ValueError):
        return None


def _handle_fitness_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("""
          SELECT
            (SELECT count(*) FROM fitness.profiles) AS profiles,
            (SELECT count(*) FROM fitness.goals) AS goals,
            (SELECT count(*) FROM fitness.foods) AS foods,
            (SELECT count(*) FROM fitness.nutrition_logs) AS nutrition_logs,
            (SELECT count(*) FROM fitness.exercises) AS exercises,
            (SELECT count(*) FROM fitness.routines) AS routines,
            (SELECT count(*) FROM fitness.workout_sessions) AS workout_sessions,
            (SELECT count(*) FROM fitness.workout_sets) AS workout_sets,
            (SELECT count(*) FROM fitness.body_metrics) AS body_metrics,
            (SELECT count(*) FROM fitness.coach_checkins) AS coach_checkins
        """, user=_user())
        return _ok(db_backend="agent_core_postgres", counts=counts)
    except Exception as exc:
        return _err(exc)


def _handle_profile_upsert(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        display_name = str(args.get("display_name") or "").strip()
        if not profile_id and not display_name:
            raise ValueError("profile_id or display_name is required")
        profile_id = profile_id or _slug("fitness-profile", display_name)
        row = sql.statement_one(f"""
          INSERT INTO fitness.profiles (profile_id, owner_id, display_name, sex, birth_date, height_cm, activity_level, timezone, preferred_units, dietary_preferences, allergies, equipment_available, injury_notes, metadata, created_at, updated_at)
          VALUES ({_q(profile_id)}, {_q(args.get('owner_id'))}, {_q(display_name or profile_id)}, {_q(args.get('sex'))}, {_q(args.get('birth_date'))}::date, {_num(args.get('height_cm'))}, {_q(args.get('activity_level') or 'moderate')}, {_q(args.get('timezone') or 'America/Caracas')}, {_j(args.get('preferred_units') or {'weight': 'kg', 'distance': 'km'})}, {_j(args.get('dietary_preferences') or [])}, {_j(args.get('allergies') or [])}, {_j(args.get('equipment_available') or [])}, {_q(args.get('injury_notes'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (profile_id) DO UPDATE SET owner_id=EXCLUDED.owner_id, display_name=EXCLUDED.display_name, sex=EXCLUDED.sex, birth_date=EXCLUDED.birth_date, height_cm=EXCLUDED.height_cm, activity_level=EXCLUDED.activity_level, timezone=EXCLUDED.timezone, preferred_units=EXCLUDED.preferred_units, dietary_preferences=EXCLUDED.dietary_preferences, allergies=EXCLUDED.allergies, equipment_available=EXCLUDED.equipment_available, injury_notes=EXCLUDED.injury_notes, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(profile=row)
    except Exception as exc:
        return _err(exc)


def _handle_goal_upsert(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        title = str(args.get("title") or args.get("goal_type") or "").strip()
        if not profile_id or not title:
            raise ValueError("profile_id and title/goal_type are required")
        goal_id = args.get("goal_id") or _slug("fitness-goal", f"{profile_id}-{title}")
        row = sql.statement_one(f"""
          INSERT INTO fitness.goals (goal_id, profile_id, goal_type, title, target_calories, target_protein_g, target_carbs_g, target_fat_g, target_fiber_g, target_water_ml, target_weight_kg, weekly_weight_delta_kg, start_date, target_date, status, metadata, created_at, updated_at)
          VALUES ({_q(goal_id)}, {_q(profile_id)}, {_q(args.get('goal_type') or 'general')}, {_q(title)}, {_num(args.get('target_calories'))}, {_num(args.get('target_protein_g'))}, {_num(args.get('target_carbs_g'))}, {_num(args.get('target_fat_g'))}, {_num(args.get('target_fiber_g'))}, {_num(args.get('target_water_ml'))}, {_num(args.get('target_weight_kg'))}, {_num(args.get('weekly_weight_delta_kg'))}, COALESCE({_q(args.get('start_date'))}::date, CURRENT_DATE), {_q(args.get('target_date'))}::date, {_q(args.get('status') or 'active')}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (goal_id) DO UPDATE SET profile_id=EXCLUDED.profile_id, goal_type=EXCLUDED.goal_type, title=EXCLUDED.title, target_calories=EXCLUDED.target_calories, target_protein_g=EXCLUDED.target_protein_g, target_carbs_g=EXCLUDED.target_carbs_g, target_fat_g=EXCLUDED.target_fat_g, target_fiber_g=EXCLUDED.target_fiber_g, target_water_ml=EXCLUDED.target_water_ml, target_weight_kg=EXCLUDED.target_weight_kg, weekly_weight_delta_kg=EXCLUDED.weekly_weight_delta_kg, start_date=EXCLUDED.start_date, target_date=EXCLUDED.target_date, status=EXCLUDED.status, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(goal=row)
    except Exception as exc:
        return _err(exc)


def _food_row_from_args(args: dict) -> dict[str, Any]:
    return {
        "source_key": args.get("source_key") or "custom",
        "external_id": args.get("external_id") or args.get("barcode"),
        "barcode": args.get("barcode"),
        "name": args.get("name"),
        "brand": args.get("brand"),
        "category": args.get("category"),
        "serving_size_g": args.get("serving_size_g"),
        "calories_per_100g": args.get("calories_per_100g"),
        "protein_g_per_100g": args.get("protein_g_per_100g"),
        "carbs_g_per_100g": args.get("carbs_g_per_100g"),
        "fat_g_per_100g": args.get("fat_g_per_100g"),
        "fiber_g_per_100g": args.get("fiber_g_per_100g"),
        "sugar_g_per_100g": args.get("sugar_g_per_100g"),
        "sodium_mg_per_100g": args.get("sodium_mg_per_100g"),
        "metadata": args.get("metadata") or {},
    }


def _handle_food_upsert(args: dict, **_kwargs) -> str:
    try:
        food = _food_row_from_args(args)
        if not str(food.get("name") or "").strip():
            raise ValueError("name is required")
        food_id = args.get("food_id") or _slug("food", f"{food.get('source_key')}-{food.get('external_id') or food.get('barcode') or food.get('name')}")
        row = sql.statement_one(f"""
          INSERT INTO fitness.foods (food_id, source_key, external_id, barcode, name, brand, category, serving_size_g, calories_per_100g, protein_g_per_100g, carbs_g_per_100g, fat_g_per_100g, fiber_g_per_100g, sugar_g_per_100g, sodium_mg_per_100g, metadata, created_at, updated_at)
          VALUES ({_q(food_id)}, {_q(food.get('source_key'))}, {_q(food.get('external_id'))}, {_q(food.get('barcode'))}, {_q(food.get('name'))}, {_q(food.get('brand'))}, {_q(food.get('category'))}, {_num(food.get('serving_size_g'))}, {_num(food.get('calories_per_100g'))}, {_num(food.get('protein_g_per_100g'))}, {_num(food.get('carbs_g_per_100g'))}, {_num(food.get('fat_g_per_100g'))}, {_num(food.get('fiber_g_per_100g'))}, {_num(food.get('sugar_g_per_100g'))}, {_num(food.get('sodium_mg_per_100g'))}, {_j(food.get('metadata') or {})}, now(), now())
          ON CONFLICT (food_id) DO UPDATE SET source_key=EXCLUDED.source_key, external_id=EXCLUDED.external_id, barcode=EXCLUDED.barcode, name=EXCLUDED.name, brand=EXCLUDED.brand, category=EXCLUDED.category, serving_size_g=EXCLUDED.serving_size_g, calories_per_100g=EXCLUDED.calories_per_100g, protein_g_per_100g=EXCLUDED.protein_g_per_100g, carbs_g_per_100g=EXCLUDED.carbs_g_per_100g, fat_g_per_100g=EXCLUDED.fat_g_per_100g, fiber_g_per_100g=EXCLUDED.fiber_g_per_100g, sugar_g_per_100g=EXCLUDED.sugar_g_per_100g, sodium_mg_per_100g=EXCLUDED.sodium_mg_per_100g, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(food=row)
    except Exception as exc:
        return _err(exc)


def _handle_food_search(args: dict, **_kwargs) -> str:
    try:
        barcode = str(args.get("barcode") or "").strip()
        query = str(args.get("query") or "").strip()
        if not barcode and not query:
            raise ValueError("query or barcode is required")
        if barcode:
            existing = sql.one(f"SELECT * FROM fitness.foods WHERE barcode={_q(barcode)} OR external_id={_q(barcode)} ORDER BY updated_at DESC", user=_user())
            if existing:
                return _ok(source="local_cache", foods=[existing])
            if args.get("external_lookup", True):
                product = _off_product(barcode)
                if product:
                    saved = json.loads(_handle_food_upsert(product))["food"]
                    return _ok(source="open_food_facts", attribution="Open Food Facts data: ODbL/DbCL; images CC BY-SA if used", foods=[saved])
        limit = int(args.get("limit") or 10)
        rows = sql.rows(f"""
          SELECT * FROM fitness.foods
          WHERE name ILIKE '%' || {_q(query)} || '%'
             OR brand ILIKE '%' || {_q(query)} || '%'
             OR barcode={_q(query)}
          ORDER BY CASE WHEN name ILIKE {_q(query + '%')} THEN 0 ELSE 1 END, updated_at DESC
          LIMIT {limit}
        """, user=_user())
        return _ok(source="local_cache", foods=rows)
    except Exception as exc:
        return _err(exc)


def _handle_nutrition_log_create(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        food_id = str(args.get("food_id") or "").strip() or None
        description = str(args.get("description") or args.get("food_name") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        if not food_id and not description:
            raise ValueError("food_id or description is required")
        nutrition_log_id = args.get("nutrition_log_id") or _slug("nutrition-log", f"{profile_id}-{args.get('occurred_at') or 'now'}-{food_id or description}")
        grams = _safe_float(args.get("grams")) or _safe_float(args.get("quantity_g"))
        row = sql.statement_one(f"""
          INSERT INTO fitness.nutrition_logs (nutrition_log_id, profile_id, food_id, occurred_at, meal_type, description, quantity, unit, grams, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, water_ml, source_confidence, notes, metadata, created_at, updated_at)
          VALUES ({_q(nutrition_log_id)}, {_q(profile_id)}, {_q(food_id)}, COALESCE({_q(args.get('occurred_at'))}::timestamptz, now()), {_q(args.get('meal_type') or 'unspecified')}, {_q(description)}, {_num(args.get('quantity'), '1')}, {_q(args.get('unit') or 'serving')}, {_num(grams)}, {_num(args.get('calories'))}, {_num(args.get('protein_g'))}, {_num(args.get('carbs_g'))}, {_num(args.get('fat_g'))}, {_num(args.get('fiber_g'))}, {_num(args.get('sugar_g'))}, {_num(args.get('sodium_mg'))}, {_num(args.get('water_ml'))}, {_num(args.get('source_confidence'), '1')}, {_q(args.get('notes'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (nutrition_log_id) DO UPDATE SET profile_id=EXCLUDED.profile_id, food_id=EXCLUDED.food_id, occurred_at=EXCLUDED.occurred_at, meal_type=EXCLUDED.meal_type, description=EXCLUDED.description, quantity=EXCLUDED.quantity, unit=EXCLUDED.unit, grams=EXCLUDED.grams, calories=EXCLUDED.calories, protein_g=EXCLUDED.protein_g, carbs_g=EXCLUDED.carbs_g, fat_g=EXCLUDED.fat_g, fiber_g=EXCLUDED.fiber_g, sugar_g=EXCLUDED.sugar_g, sodium_mg=EXCLUDED.sodium_mg, water_ml=EXCLUDED.water_ml, source_confidence=EXCLUDED.source_confidence, notes=EXCLUDED.notes, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(nutrition_log=row)
    except Exception as exc:
        return _err(exc)


def _handle_nutrition_day_summary(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        date = args.get("date")
        if not profile_id:
            raise ValueError("profile_id is required")
        summary = sql.one(f"""
          SELECT COALESCE(sum(calories),0) AS calories, COALESCE(sum(protein_g),0) AS protein_g, COALESCE(sum(carbs_g),0) AS carbs_g,
                 COALESCE(sum(fat_g),0) AS fat_g, COALESCE(sum(fiber_g),0) AS fiber_g, COALESCE(sum(water_ml),0) AS water_ml,
                 count(*) AS entries
          FROM fitness.nutrition_logs
          WHERE profile_id={_q(profile_id)} AND occurred_at::date=COALESCE({_q(date)}::date, CURRENT_DATE)
        """, user=_user())
        meals = sql.rows(f"""
          SELECT meal_type, COALESCE(sum(calories),0) AS calories, count(*) AS entries
          FROM fitness.nutrition_logs
          WHERE profile_id={_q(profile_id)} AND occurred_at::date=COALESCE({_q(date)}::date, CURRENT_DATE)
          GROUP BY meal_type ORDER BY meal_type
        """, user=_user())
        goal = sql.one(f"SELECT * FROM fitness.goals WHERE profile_id={_q(profile_id)} AND status='active' ORDER BY start_date DESC, updated_at DESC", user=_user())
        return _ok(date=date, profile_id=profile_id, summary=summary, meals=meals, active_goal=goal)
    except Exception as exc:
        return _err(exc)


def _handle_exercise_upsert(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        exercise_id = args.get("exercise_id") or _slug("exercise", f"{args.get('source_key') or 'custom'}-{args.get('external_id') or name}")
        row = sql.statement_one(f"""
          INSERT INTO fitness.exercises (exercise_id, source_key, external_id, name, aliases, category, movement_pattern, primary_muscles, secondary_muscles, equipment, difficulty, instructions, media_refs, contraindications, metadata, created_at, updated_at)
          VALUES ({_q(exercise_id)}, {_q(args.get('source_key') or 'custom')}, {_q(args.get('external_id'))}, {_q(name)}, {_j(args.get('aliases') or [])}, {_q(args.get('category'))}, {_q(args.get('movement_pattern'))}, {_j(args.get('primary_muscles') or [])}, {_j(args.get('secondary_muscles') or [])}, {_j(args.get('equipment') or [])}, {_q(args.get('difficulty'))}, {_j(args.get('instructions') or [])}, {_j(args.get('media_refs') or [])}, {_j(args.get('contraindications') or [])}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (exercise_id) DO UPDATE SET source_key=EXCLUDED.source_key, external_id=EXCLUDED.external_id, name=EXCLUDED.name, aliases=EXCLUDED.aliases, category=EXCLUDED.category, movement_pattern=EXCLUDED.movement_pattern, primary_muscles=EXCLUDED.primary_muscles, secondary_muscles=EXCLUDED.secondary_muscles, equipment=EXCLUDED.equipment, difficulty=EXCLUDED.difficulty, instructions=EXCLUDED.instructions, media_refs=EXCLUDED.media_refs, contraindications=EXCLUDED.contraindications, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(exercise=row)
    except Exception as exc:
        return _err(exc)


def _handle_exercise_search(args: dict, **_kwargs) -> str:
    try:
        query = str(args.get("query") or "").strip()
        muscle = str(args.get("muscle") or "").strip()
        equipment = str(args.get("equipment") or "").strip()
        limit = int(args.get("limit") or 20)
        rows = sql.rows(f"""
          SELECT * FROM fitness.exercises
          WHERE ({_q(query)} IS NULL OR {_q(query)} = '' OR name ILIKE '%' || {_q(query)} || '%' OR aliases::text ILIKE '%' || {_q(query)} || '%')
            AND ({_q(muscle)} IS NULL OR {_q(muscle)} = '' OR primary_muscles::text ILIKE '%' || {_q(muscle)} || '%' OR secondary_muscles::text ILIKE '%' || {_q(muscle)} || '%')
            AND ({_q(equipment)} IS NULL OR {_q(equipment)} = '' OR equipment::text ILIKE '%' || {_q(equipment)} || '%')
          ORDER BY name LIMIT {limit}
        """, user=_user())
        return _ok(exercises=rows)
    except Exception as exc:
        return _err(exc)


def _handle_routine_create(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        title = str(args.get("title") or "").strip()
        days = args.get("days") or []
        if not profile_id or not title:
            raise ValueError("profile_id and title are required")
        if not isinstance(days, list):
            raise ValueError("days must be a list")
        routine_id = args.get("routine_id") or _slug("routine", f"{profile_id}-{title}")
        routine = sql.statement_one(f"""
          INSERT INTO fitness.routines (routine_id, profile_id, title, goal_type, split_type, days_per_week, status, plan, metadata, created_at, updated_at)
          VALUES ({_q(routine_id)}, {_q(profile_id)}, {_q(title)}, {_q(args.get('goal_type') or 'general')}, {_q(args.get('split_type'))}, {_int(args.get('days_per_week') or len(days) or None)}, {_q(args.get('status') or 'active')}, {_j(args.get('plan') or {})}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (routine_id) DO UPDATE SET profile_id=EXCLUDED.profile_id, title=EXCLUDED.title, goal_type=EXCLUDED.goal_type, split_type=EXCLUDED.split_type, days_per_week=EXCLUDED.days_per_week, status=EXCLUDED.status, plan=EXCLUDED.plan, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        sql.psql(f"DELETE FROM fitness.routine_days WHERE routine_id={_q(routine_id)};", user=_user())
        saved_days = []
        saved_exercises = []
        for day_index, day in enumerate(days, start=1):
            day_row = sql.statement_one(f"""
              INSERT INTO fitness.routine_days (routine_id, day_index, title, notes, metadata)
              VALUES ({_q(routine_id)}, {int(day.get('day_index') or day_index)}, {_q(day.get('name') or day.get('title') or f'Day {day_index}')}, {_q(day.get('notes'))}, {_j(day.get('metadata') or {})})
              RETURNING *
            """, user=_user())
            saved_days.append(day_row)
            routine_day_id = day_row.get("routine_day_id") if isinstance(day_row, dict) else None
            for order_index, exercise in enumerate(day.get("exercises") or [], start=1):
                saved_exercises.append(sql.statement_one(f"""
                  INSERT INTO fitness.routine_exercises (routine_day_id, exercise_id, order_index, sets, target_reps_min, target_reps_max, target_rpe, target_rir, rest_seconds, tempo, progression_rule, notes, metadata)
                  VALUES ({_q(routine_day_id)}, {_q(exercise.get('exercise_id'))}, {int(exercise.get('order_index') or order_index)}, {_int(exercise.get('sets'), 'NULL')}, {_int(exercise.get('target_reps_min'), 'NULL')}, {_int(exercise.get('target_reps_max'), 'NULL')}, {_num(exercise.get('target_rpe'))}, {_num(exercise.get('target_rir'))}, {_int(exercise.get('rest_seconds'), 'NULL')}, {_q(exercise.get('tempo'))}, {_j(exercise.get('progression_rule') or {})}, {_q(exercise.get('notes'))}, {_j(exercise.get('metadata') or {})})
                  RETURNING *
                """, user=_user()))
        return _ok(routine=routine, days=saved_days, exercises=saved_exercises)
    except Exception as exc:
        return _err(exc)


def _handle_routine_get(args: dict, **_kwargs) -> str:
    try:
        routine_id = str(args.get("routine_id") or "").strip()
        if not routine_id:
            raise ValueError("routine_id is required")
        routine = sql.one(f"SELECT * FROM fitness.routines WHERE routine_id={_q(routine_id)}", user=_user())
        days = sql.rows(f"SELECT * FROM fitness.routine_days WHERE routine_id={_q(routine_id)} ORDER BY day_index", user=_user())
        exercises = sql.rows(f"""
          SELECT re.*, e.name AS exercise_name
          FROM fitness.routine_exercises re LEFT JOIN fitness.exercises e ON e.exercise_id=re.exercise_id
          WHERE re.routine_day_id IN (SELECT routine_day_id FROM fitness.routine_days WHERE routine_id={_q(routine_id)})
          ORDER BY re.routine_day_id, re.order_index
        """, user=_user())
        return _ok(routine=routine, days=days, exercises=exercises)
    except Exception as exc:
        return _err(exc)


def _handle_workout_session_create(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        session_id = args.get("session_id") or _slug("workout", f"{profile_id}-{args.get('started_at') or 'now'}-{args.get('title') or args.get('activity_type') or 'session'}")
        row = sql.statement_one(f"""
          INSERT INTO fitness.workout_sessions (session_id, profile_id, routine_id, routine_day_id, started_at, ended_at, activity_type, title, status, duration_minutes, distance_km, calories_burned, perceived_effort, readiness_score, notes, metadata, created_at, updated_at)
          VALUES ({_q(session_id)}, {_q(profile_id)}, {_q(args.get('routine_id'))}, {_q(args.get('routine_day_id'))}, COALESCE({_q(args.get('started_at'))}::timestamptz, now()), {_q(args.get('ended_at'))}::timestamptz, {_q(args.get('activity_type') or 'strength')}, {_q(args.get('title'))}, {_q(args.get('status') or 'in_progress')}, {_num(args.get('duration_minutes'))}, {_num(args.get('distance_km'))}, {_num(args.get('calories_burned'))}, {_num(args.get('perceived_effort'))}, {_num(args.get('readiness_score'))}, {_q(args.get('notes'))}, {_j(args.get('metadata') or {})}, now(), now())
          ON CONFLICT (session_id) DO UPDATE SET routine_id=EXCLUDED.routine_id, routine_day_id=EXCLUDED.routine_day_id, started_at=EXCLUDED.started_at, ended_at=EXCLUDED.ended_at, activity_type=EXCLUDED.activity_type, title=EXCLUDED.title, status=EXCLUDED.status, duration_minutes=EXCLUDED.duration_minutes, distance_km=EXCLUDED.distance_km, calories_burned=EXCLUDED.calories_burned, perceived_effort=EXCLUDED.perceived_effort, readiness_score=EXCLUDED.readiness_score, notes=EXCLUDED.notes, metadata=EXCLUDED.metadata, updated_at=now()
          RETURNING *
        """, user=_user())
        return _ok(session=row)
    except Exception as exc:
        return _err(exc)


def _handle_workout_set_log(args: dict, **_kwargs) -> str:
    try:
        session_id = str(args.get("session_id") or "").strip()
        exercise_id = str(args.get("exercise_id") or "").strip()
        if not session_id or not exercise_id:
            raise ValueError("session_id and exercise_id are required")
        reps = int(args.get("reps") or 0) if args.get("reps") not in (None, "") else None
        weight = _safe_float(args.get("weight_kg"))
        volume = _money((weight or 0) * (reps or 0))
        est_1rm = _estimated_1rm(weight, reps)
        row = sql.statement_one(f"""
          INSERT INTO fitness.workout_sets (session_id, exercise_id, set_index, set_type, weight_kg, reps, duration_seconds, distance_m, rpe, rir, completed, failure, rest_seconds, volume_load, estimated_1rm, notes, metadata)
          VALUES ({_q(session_id)}, {_q(exercise_id)}, {_int(args.get('set_index'), '1')}, {_q(args.get('set_type') or 'working')}, {_num(weight)}, {_int(reps)}, {_int(args.get('duration_seconds'), 'NULL')}, {_num(args.get('distance_m'))}, {_num(args.get('rpe'))}, {_num(args.get('rir'))}, {_q(args.get('completed', True))}, {_q(args.get('failure', False))}, {_int(args.get('rest_seconds'), 'NULL')}, {_num(volume, '0')}, {_num(est_1rm)}, {_q(args.get('notes'))}, {_j(args.get('metadata') or {})})
          RETURNING *
        """, user=_user())
        return _ok(set=row, computed={"volume_load": volume, "estimated_1rm": est_1rm})
    except Exception as exc:
        return _err(exc)


def _handle_workout_finish(args: dict, **_kwargs) -> str:
    try:
        session_id = str(args.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        row = sql.statement_one(f"""
          UPDATE fitness.workout_sessions
          SET ended_at=COALESCE({_q(args.get('ended_at'))}::timestamptz, now()), status={_q(args.get('status') or 'completed')}, duration_minutes=COALESCE({_num(args.get('duration_minutes'))}, duration_minutes), perceived_effort=COALESCE({_num(args.get('perceived_effort'))}, perceived_effort), notes=COALESCE({_q(args.get('notes'))}, notes), updated_at=now()
          WHERE session_id={_q(session_id)}
          RETURNING *
        """, user=_user())
        sets = sql.rows(f"SELECT exercise_id, count(*) AS sets, sum(volume_load) AS volume_load, max(estimated_1rm) AS best_estimated_1rm FROM fitness.workout_sets WHERE session_id={_q(session_id)} GROUP BY exercise_id ORDER BY exercise_id", user=_user())
        return _ok(session=row, exercise_summaries=sets)
    except Exception as exc:
        return _err(exc)


def _handle_body_metric_log_create(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        row = sql.statement_one(f"""
          INSERT INTO fitness.body_metrics (profile_id, measured_at, weight_kg, body_fat_pct, waist_cm, chest_cm, hip_cm, arm_cm, thigh_cm, neck_cm, resting_hr_bpm, sleep_hours, mood, energy_level, stress_level, source, metadata)
          VALUES ({_q(profile_id)}, COALESCE({_q(args.get('measured_at'))}::timestamptz, now()), {_num(args.get('weight_kg'))}, {_num(args.get('body_fat_pct'))}, {_num(args.get('waist_cm'))}, {_num(args.get('chest_cm'))}, {_num(args.get('hip_cm'))}, {_num(args.get('arm_cm'))}, {_num(args.get('thigh_cm'))}, {_num(args.get('neck_cm'))}, {_num(args.get('resting_hr_bpm'))}, {_num(args.get('sleep_hours'))}, {_q(args.get('mood'))}, {_num(args.get('energy_level'))}, {_num(args.get('stress_level'))}, {_q(args.get('source') or 'manual')}, {_j(args.get('metadata') or {})})
          RETURNING *
        """, user=_user())
        return _ok(body_metric=row)
    except Exception as exc:
        return _err(exc)


def _handle_checkin_create(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        summary = str(args.get("summary") or "").strip()
        if not profile_id or not summary:
            raise ValueError("profile_id and summary are required")
        checkin_id = args.get("checkin_id") or _slug("fitness-checkin", f"{profile_id}-{args.get('occurred_at') or 'now'}")
        row = sql.statement_one(f"""
          INSERT INTO fitness.coach_checkins (checkin_id, profile_id, occurred_at, summary, sleep_quality, soreness, stress_level, hunger_level, energy_level, adherence_score, blockers, next_steps, metadata, created_at)
          VALUES ({_q(checkin_id)}, {_q(profile_id)}, COALESCE({_q(args.get('occurred_at'))}::timestamptz, now()), {_q(summary)}, {_num(args.get('sleep_quality'))}, {_num(args.get('soreness'))}, {_num(args.get('stress_level'))}, {_num(args.get('hunger_level'))}, {_num(args.get('energy_level'))}, {_num(args.get('adherence_score'))}, {_q(args.get('blockers'))}, {_q(args.get('next_steps'))}, {_j(args.get('metadata') or {})}, now())
          ON CONFLICT (checkin_id) DO UPDATE SET summary=EXCLUDED.summary, sleep_quality=EXCLUDED.sleep_quality, soreness=EXCLUDED.soreness, stress_level=EXCLUDED.stress_level, hunger_level=EXCLUDED.hunger_level, energy_level=EXCLUDED.energy_level, adherence_score=EXCLUDED.adherence_score, blockers=EXCLUDED.blockers, next_steps=EXCLUDED.next_steps, metadata=EXCLUDED.metadata
          RETURNING *
        """, user=_user())
        return _ok(checkin=row)
    except Exception as exc:
        return _err(exc)


def _handle_progress_summary(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        start_date = args.get("start_date")
        end_date = args.get("end_date")
        summary = sql.one(f"""
          SELECT count(DISTINCT nl.occurred_at::date) AS logged_nutrition_days,
                 avg(nl.calories) AS avg_calories,
                 avg(nl.protein_g) AS avg_protein_g,
                 (SELECT weight_kg FROM fitness.body_metrics bm WHERE bm.profile_id={_q(profile_id)} AND ({_q(end_date)} IS NULL OR bm.measured_at::date <= {_q(end_date)}::date) ORDER BY bm.measured_at DESC LIMIT 1) AS latest_weight_kg,
                 (SELECT weight_kg FROM fitness.body_metrics bm WHERE bm.profile_id={_q(profile_id)} AND ({_q(start_date)} IS NULL OR bm.measured_at::date >= {_q(start_date)}::date) ORDER BY bm.measured_at ASC LIMIT 1) AS starting_weight_kg
          FROM fitness.nutrition_logs nl
          WHERE nl.profile_id={_q(profile_id)}
            AND ({_q(start_date)} IS NULL OR nl.occurred_at::date >= {_q(start_date)}::date)
            AND ({_q(end_date)} IS NULL OR nl.occurred_at::date <= {_q(end_date)}::date)
        """, user=_user())
        training = sql.rows(f"""
          SELECT activity_type, count(*) AS sessions, sum(duration_minutes) AS minutes, sum(calories_burned) AS calories_burned
          FROM fitness.workout_sessions
          WHERE profile_id={_q(profile_id)}
            AND ({_q(start_date)} IS NULL OR started_at::date >= {_q(start_date)}::date)
            AND ({_q(end_date)} IS NULL OR started_at::date <= {_q(end_date)}::date)
          GROUP BY activity_type ORDER BY sessions DESC
        """, user=_user())
        strength = sql.rows(f"""
          SELECT e.name AS exercise_name, max(ws.estimated_1rm) AS best_estimated_1rm, sum(ws.volume_load) AS volume_load
          FROM fitness.workout_sets ws
          JOIN fitness.workout_sessions s ON s.session_id=ws.session_id
          LEFT JOIN fitness.exercises e ON e.exercise_id=ws.exercise_id
          WHERE s.profile_id={_q(profile_id)}
            AND ({_q(start_date)} IS NULL OR s.started_at::date >= {_q(start_date)}::date)
            AND ({_q(end_date)} IS NULL OR s.started_at::date <= {_q(end_date)}::date)
          GROUP BY e.name ORDER BY volume_load DESC NULLS LAST LIMIT 10
        """, user=_user())
        return _ok(profile_id=profile_id, range={"start_date": start_date, "end_date": end_date}, summary=summary, training=training, strength=strength)
    except Exception as exc:
        return _err(exc)


def _handle_coach_review(args: dict, **_kwargs) -> str:
    try:
        profile_id = str(args.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        progress = json.loads(_handle_progress_summary(args))
        target = sql.one(f"SELECT * FROM fitness.goals WHERE profile_id={_q(profile_id)} AND status='active' ORDER BY start_date DESC, updated_at DESC", user=_user())
        recommendations = []
        summary = progress.get("summary") or {}
        avg_calories = float(summary.get("avg_calories") or 0)
        if target and target.get("target_calories") and avg_calories:
            delta = avg_calories - float(target["target_calories"])
            if abs(delta) > 150:
                recommendations.append({"category": "nutrition", "recommendation": "Ajustar adherencia calórica diaria", "rationale": f"Promedio estimado {avg_calories:.0f} kcal vs objetivo {float(target['target_calories']):.0f} kcal."})
        sessions = sum(int(row.get("sessions") or 0) for row in progress.get("training") or [])
        if sessions < int(args.get("minimum_weekly_sessions") or 3):
            recommendations.append({"category": "training", "recommendation": "Programar sesiones más cortas pero cumplibles", "rationale": f"Se registraron {sessions} sesiones en el rango."})
        if not recommendations:
            recommendations.append({"category": "coaching", "recommendation": "Mantener plan actual y seguir registrando datos", "rationale": "No hay señales estructuradas suficientes para cambiar el plan."})
        row = sql.statement_one(f"""
          INSERT INTO fitness.coach_recommendations (profile_id, category, recommendation, rationale, status, metadata)
          VALUES ({_q(profile_id)}, 'review', {_q('Revisión de progreso generada')}, {_q('Sugerencias estructuradas calculadas desde logs de Fitness Core')}, 'generated', {_j({'recommendations': recommendations, 'progress': progress})})
          RETURNING *
        """, user=_user())
        return _ok(progress=progress, active_goal=target, recommendations=recommendations, recommendation_event=row, safety_note="Coaching no médico: ante dolor agudo, lesiones, enfermedades, embarazo o señales de trastorno alimentario, escalar a profesional de salud.")
    except Exception as exc:
        return _err(exc)


# Registry
registry.register(name="fitness_status", toolset="fitness", schema=_schema("fitness_status", "Return Fitness Coach Core row counts and DB backend.", {}), handler=_handle_fitness_status, check_fn=_check_fitness, emoji="💪")
registry.register(name="fitness_profile_upsert", toolset="fitness", schema=_schema("fitness_profile_upsert", "Create/update a personal fitness profile with height, preferences, equipment, allergies, and constraints.", {"profile_id": {"type": "string"}, "owner_id": {"type": "string"}, "display_name": {"type": "string"}, "sex": {"type": "string"}, "birth_date": {"type": "string"}, "height_cm": {"type": "number"}, "activity_level": {"type": "string"}, "timezone": {"type": "string"}, "preferred_units": {"type": "object"}, "dietary_preferences": {"type": "array", "items": {"type": "string"}}, "allergies": {"type": "array", "items": {"type": "string"}}, "equipment_available": {"type": "array", "items": {"type": "string"}}, "injury_notes": {"type": "string"}, **_meta_props()}), handler=_handle_profile_upsert, check_fn=_check_fitness, emoji="💪")
registry.register(name="fitness_goal_upsert", toolset="fitness", schema=_schema("fitness_goal_upsert", "Create/update calorie, macro, body-weight, hydration, or training goals for a profile.", {"goal_id": {"type": "string"}, "profile_id": {"type": "string"}, "goal_type": {"type": "string"}, "title": {"type": "string"}, "target_calories": {"type": "number"}, "target_protein_g": {"type": "number"}, "target_carbs_g": {"type": "number"}, "target_fat_g": {"type": "number"}, "target_fiber_g": {"type": "number"}, "target_water_ml": {"type": "number"}, "target_weight_kg": {"type": "number"}, "weekly_weight_delta_kg": {"type": "number"}, "start_date": {"type": "string"}, "target_date": {"type": "string"}, "status": {"type": "string"}, **_meta_props()}, ["profile_id"]), handler=_handle_goal_upsert, check_fn=_check_fitness, emoji="🎯")
registry.register(name="fitness_food_upsert", toolset="fitness", schema=_schema("fitness_food_upsert", "Create/update a custom or cached food with per-100g nutrition values.", {"food_id": {"type": "string"}, "source_key": {"type": "string"}, "external_id": {"type": "string"}, "barcode": {"type": "string"}, "name": {"type": "string"}, "brand": {"type": "string"}, "category": {"type": "string"}, "serving_size_g": {"type": "number"}, "calories_per_100g": {"type": "number"}, "protein_g_per_100g": {"type": "number"}, "carbs_g_per_100g": {"type": "number"}, "fat_g_per_100g": {"type": "number"}, "fiber_g_per_100g": {"type": "number"}, "sugar_g_per_100g": {"type": "number"}, "sodium_mg_per_100g": {"type": "number"}, **_meta_props()}, ["name"]), handler=_handle_food_upsert, check_fn=_check_fitness, emoji="🍎")
registry.register(name="fitness_food_search", toolset="fitness", schema=_schema("fitness_food_search", "Search local food cache, or lookup/cache an Open Food Facts product by barcode.", {"query": {"type": "string"}, "barcode": {"type": "string"}, "external_lookup": {"type": "boolean"}, "limit": {"type": "integer"}}), handler=_handle_food_search, check_fn=_check_fitness, emoji="🍎")
registry.register(name="fitness_nutrition_log_create", toolset="fitness", schema=_schema("fitness_nutrition_log_create", "Log a meal/food entry with calories/macros snapshot for historical nutrition tracking.", {"nutrition_log_id": {"type": "string"}, "profile_id": {"type": "string"}, "food_id": {"type": "string"}, "occurred_at": {"type": "string"}, "meal_type": {"type": "string"}, "description": {"type": "string"}, "quantity": {"type": "number"}, "unit": {"type": "string"}, "grams": {"type": "number"}, "calories": {"type": "number"}, "protein_g": {"type": "number"}, "carbs_g": {"type": "number"}, "fat_g": {"type": "number"}, "fiber_g": {"type": "number"}, "sugar_g": {"type": "number"}, "sodium_mg": {"type": "number"}, "water_ml": {"type": "number"}, "notes": {"type": "string"}, **_meta_props()}, ["profile_id"]), handler=_handle_nutrition_log_create, check_fn=_check_fitness, emoji="🍽️")
registry.register(name="fitness_nutrition_day_summary", toolset="fitness", schema=_schema("fitness_nutrition_day_summary", "Summarize a profile's calories, macros, water, meals, and active goal for one day.", {"profile_id": {"type": "string"}, "date": {"type": "string"}}, ["profile_id"]), handler=_handle_nutrition_day_summary, check_fn=_check_fitness, emoji="📊")
registry.register(name="fitness_exercise_upsert", toolset="fitness", schema=_schema("fitness_exercise_upsert", "Create/update an exercise with muscles, equipment, instructions, media references, and contraindications.", {"exercise_id": {"type": "string"}, "source_key": {"type": "string"}, "external_id": {"type": "string"}, "name": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}}, "category": {"type": "string"}, "movement_pattern": {"type": "string"}, "primary_muscles": {"type": "array", "items": {"type": "string"}}, "secondary_muscles": {"type": "array", "items": {"type": "string"}}, "equipment": {"type": "array", "items": {"type": "string"}}, "difficulty": {"type": "string"}, "instructions": {"type": "array", "items": {"type": "string"}}, "media_refs": {"type": "array", "items": {"type": "object"}}, "contraindications": {"type": "array", "items": {"type": "string"}}, **_meta_props()}, ["name"]), handler=_handle_exercise_upsert, check_fn=_check_fitness, emoji="🏋️")
registry.register(name="fitness_exercise_search", toolset="fitness", schema=_schema("fitness_exercise_search", "Search exercises by text, target muscle, or available equipment.", {"query": {"type": "string"}, "muscle": {"type": "string"}, "equipment": {"type": "string"}, "limit": {"type": "integer"}}), handler=_handle_exercise_search, check_fn=_check_fitness, emoji="🏋️")
registry.register(name="fitness_routine_create", toolset="fitness", schema=_schema("fitness_routine_create", "Create/update a routine template with days and exercise prescriptions including sets, rep ranges, RPE/RIR, rest, and progression rules.", {"routine_id": {"type": "string"}, "profile_id": {"type": "string"}, "title": {"type": "string"}, "goal_type": {"type": "string"}, "split_type": {"type": "string"}, "days_per_week": {"type": "integer"}, "status": {"type": "string"}, "plan": {"type": "object"}, "days": {"type": "array", "items": {"type": "object"}}, **_meta_props()}, ["profile_id", "title"]), handler=_handle_routine_create, check_fn=_check_fitness, emoji="📋")
registry.register(name="fitness_routine_get", toolset="fitness", schema=_schema("fitness_routine_get", "Read a routine with its days and exercise prescriptions.", {"routine_id": {"type": "string"}}, ["routine_id"]), handler=_handle_routine_get, check_fn=_check_fitness, emoji="📋")
registry.register(name="fitness_workout_session_create", toolset="fitness", schema=_schema("fitness_workout_session_create", "Start or upsert a workout/cardio/activity session.", {"session_id": {"type": "string"}, "profile_id": {"type": "string"}, "routine_id": {"type": "string"}, "routine_day_id": {"type": "string"}, "started_at": {"type": "string"}, "ended_at": {"type": "string"}, "activity_type": {"type": "string"}, "title": {"type": "string"}, "status": {"type": "string"}, "duration_minutes": {"type": "number"}, "distance_km": {"type": "number"}, "calories_burned": {"type": "number"}, "perceived_effort": {"type": "number"}, "readiness_score": {"type": "number"}, "notes": {"type": "string"}, **_meta_props()}, ["profile_id"]), handler=_handle_workout_session_create, check_fn=_check_fitness, emoji="🏃")
registry.register(name="fitness_workout_set_log", toolset="fitness", schema=_schema("fitness_workout_set_log", "Log a set with weight/reps/RPE/RIR/failure and computed volume/estimated 1RM.", {"session_id": {"type": "string"}, "exercise_id": {"type": "string"}, "set_index": {"type": "integer"}, "set_type": {"type": "string"}, "weight_kg": {"type": "number"}, "reps": {"type": "integer"}, "duration_seconds": {"type": "integer"}, "distance_m": {"type": "number"}, "rpe": {"type": "number"}, "rir": {"type": "number"}, "completed": {"type": "boolean"}, "failure": {"type": "boolean"}, "rest_seconds": {"type": "integer"}, "notes": {"type": "string"}, **_meta_props()}, ["session_id", "exercise_id"]), handler=_handle_workout_set_log, check_fn=_check_fitness, emoji="🏋️")
registry.register(name="fitness_workout_finish", toolset="fitness", schema=_schema("fitness_workout_finish", "Finish a workout session and return exercise summaries.", {"session_id": {"type": "string"}, "ended_at": {"type": "string"}, "status": {"type": "string"}, "duration_minutes": {"type": "number"}, "perceived_effort": {"type": "number"}, "notes": {"type": "string"}}, ["session_id"]), handler=_handle_workout_finish, check_fn=_check_fitness, emoji="✅")
registry.register(name="fitness_body_metric_log_create", toolset="fitness", schema=_schema("fitness_body_metric_log_create", "Log body weight, measurements, sleep, mood, energy, stress, and other health metrics.", {"profile_id": {"type": "string"}, "measured_at": {"type": "string"}, "weight_kg": {"type": "number"}, "body_fat_pct": {"type": "number"}, "waist_cm": {"type": "number"}, "chest_cm": {"type": "number"}, "hip_cm": {"type": "number"}, "arm_cm": {"type": "number"}, "thigh_cm": {"type": "number"}, "neck_cm": {"type": "number"}, "resting_hr_bpm": {"type": "number"}, "sleep_hours": {"type": "number"}, "mood": {"type": "string"}, "energy_level": {"type": "number"}, "stress_level": {"type": "number"}, "source": {"type": "string"}, **_meta_props()}, ["profile_id"]), handler=_handle_body_metric_log_create, check_fn=_check_fitness, emoji="⚖️")
registry.register(name="fitness_checkin_create", toolset="fitness", schema=_schema("fitness_checkin_create", "Record a subjective coach check-in: sleep, soreness, stress, hunger, energy, adherence, blockers, next steps.", {"checkin_id": {"type": "string"}, "profile_id": {"type": "string"}, "occurred_at": {"type": "string"}, "summary": {"type": "string"}, "sleep_quality": {"type": "number"}, "soreness": {"type": "number"}, "stress_level": {"type": "number"}, "hunger_level": {"type": "number"}, "energy_level": {"type": "number"}, "adherence_score": {"type": "number"}, "blockers": {"type": "string"}, "next_steps": {"type": "string"}, **_meta_props()}, ["profile_id", "summary"]), handler=_handle_checkin_create, check_fn=_check_fitness, emoji="🧠")
registry.register(name="fitness_progress_summary", toolset="fitness", schema=_schema("fitness_progress_summary", "Summarize nutrition, body weight trend, training sessions, volume, and strength highlights for a date range.", {"profile_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, ["profile_id"]), handler=_handle_progress_summary, check_fn=_check_fitness, emoji="📈")
registry.register(name="fitness_coach_review", toolset="fitness", schema=_schema("fitness_coach_review", "Generate a structured non-medical coaching review from logs, goals, adherence, and training frequency.", {"profile_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}, "minimum_weekly_sessions": {"type": "integer"}}, ["profile_id"]), handler=_handle_coach_review, check_fn=_check_fitness, emoji="🧑‍🏫")
