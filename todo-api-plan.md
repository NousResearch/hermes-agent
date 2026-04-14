# Todo REST API Plan

## Overview

A modern JSON REST API for managing user-owned todos. Each todo belongs to a single user. Users can create, read, update, delete, filter, and paginate their own todos.

**Base URL:** `/api/v1`  
**Content-Type:** `application/json`

---

## Data Model

### Todo

| Field         | Type       | Required | Description                               |
|---------------|------------|----------|-------------------------------------------|
| `id`          | `string`   | auto     | UUID, primary key                         |
| `title`       | `string`   | yes      | 1–200 characters                          |
| `description` | `string`   | no       | Max 2000 characters                       |
| `completed`   | `boolean`  | auto     | Default `false`                           |
| `priority`    | `enum`     | no       | `low`, `medium`, `high`. Default `medium` |
| `due_date`    | `string`   | no       | ISO 8601 date (`YYYY-MM-DD`)              |
| `created_at`  | `string`   | auto     | ISO 8601 timestamp                        |
| `updated_at`  | `string`   | auto     | ISO 8601 timestamp                        |
| `user_id`     | `string`   | auto     | Owner's user ID (not exposed in responses)|

### User

| Field         | Type     | Description              |
|---------------|----------|--------------------------|
| `id`          | `string` | UUID, primary key        |
| `email`       | `string` | Unique, used for login   |
| `password`    | `string` | Hashed (bcrypt), write-only |
| `created_at`  | `string` | ISO 8601 timestamp       |

---

## Authentication & Authorization

### Mechanism

- **JWT Bearer tokens** issued via a login endpoint.
- Tokens contain `sub` (user ID) and `exp` (expiration).
- Access token TTL: 15 minutes. Refresh token TTL: 7 days.

### Endpoints

| Method | Path                | Auth  | Description                  |
|--------|---------------------|-------|------------------------------|
| POST   | `/auth/register`    | none  | Create account, return tokens|
| POST   | `/auth/login`       | none  | Email + password → tokens    |
| POST   | `/auth/refresh`     | none  | Refresh token → new tokens   |

### Authorization Rules

- All `/todos` endpoints require a valid access token (`Authorization: Bearer <token>`).
- Users can only access todos where `user_id` matches the authenticated user's ID.
- Attempting to access another user's todo returns `404` (not `403`) to avoid leaking existence.

---

## Endpoints

### Todos

| Method | Path             | Description                                        |
|--------|------------------|----------------------------------------------------|
| GET    | `/todos`         | List user's todos (with filtering & pagination)    |
| POST   | `/todos`         | Create a new todo                                  |
| GET    | `/todos/:id`     | Get a single todo by ID                            |
| PATCH  | `/todos/:id`     | Partially update a todo                            |
| DELETE | `/todos/:id`     | Delete a todo                                      |
| PATCH  | `/todos/:id/toggle` | Shortcut: toggle `completed` to the opposite value |

### Request Details

#### `POST /todos`

```json
{
  "title": "Buy groceries",
  "description": "Milk, eggs, bread",
  "priority": "high",
  "due_date": "2026-04-20"
}
```

#### `PATCH /todos/:id`

All fields optional (at least one required):

```json
{
  "title": "Buy groceries (urgent)",
  "completed": true
}
```

#### `GET /todos` — Query Parameters

| Parameter    | Type    | Default  | Description                          |
|--------------|---------|----------|--------------------------------------|
| `completed`  | boolean | —        | Filter by completion status          |
| `priority`   | string  | —        | Filter by priority level             |
| `search`     | string  | —        | Case-insensitive search on title     |
| `sort_by`    | string  | `created_at` | Sort field: `created_at`, `due_date`, `priority` |
| `order`      | string  | `desc`   | Sort order: `asc` or `desc`          |
| `page`       | int     | `1`      | Page number (1-indexed)              |
| `per_page`   | int     | `20`     | Items per page (max 100)             |

#### Response — `GET /todos`

```json
{
  "data": [ { /* todo object */ } ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 53,
    "total_pages": 3
  }
}
```

#### Response — Single Todo (GET/PATCH/POST)

```json
{
  "data": { /* todo object */ }
}
```

#### Response — `DELETE /todos/:id`

```
204 No Content
```

---

## Error Handling

### Error Response Schema

All errors follow a consistent structure:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more fields are invalid.",
    "details": [
      { "field": "title", "message": "Title is required." },
      { "field": "due_date", "message": "Must be a valid ISO 8601 date." }
    ]
  }
}
```

| Field           | Type     | Description                              |
|-----------------|----------|------------------------------------------|
| `error.code`    | string   | Machine-readable error code              |
| `error.message` | string   | Human-readable summary                   |
| `error.details` | array    | Optional. Field-level validation errors  |

### HTTP Status Codes

| Code | Meaning                  | When Used                                           |
|------|--------------------------|-----------------------------------------------------|
| 200  | OK                       | Successful GET, PATCH                               |
| 201  | Created                  | Successful POST                                     |
| 204  | No Content               | Successful DELETE                                   |
| 400  | Bad Request              | Malformed JSON, invalid query params                |
| 401  | Unauthorized             | Missing or invalid/expired token                    |
| 404  | Not Found                | Todo does not exist **or** belongs to another user  |
| 409  | Conflict                 | Duplicate resource (e.g., email already registered) |
| 422  | Unprocessable Entity     | Validation errors on request body                   |
| 429  | Too Many Requests        | Rate limit exceeded                                 |
| 500  | Internal Server Error    | Unexpected server failure                           |

### Error Codes Reference

| Code                      | HTTP | Description                        |
|---------------------------|------|------------------------------------|
| `VALIDATION_ERROR`        | 422  | Request body failed validation     |
| `AUTH_REQUIRED`           | 401  | No token provided                  |
| `AUTH_INVALID`            | 401  | Token is invalid or expired        |
| `NOT_FOUND`               | 404  | Resource not found                 |
| `CONFLICT`                | 409  | Duplicate resource                 |
| `RATE_LIMITED`            | 429  | Too many requests                  |
| `INTERNAL_ERROR`          | 500  | Unexpected server error            |

---

## Rate Limiting

- **Limit:** 100 requests per minute per authenticated user (60 req/min for unauthenticated auth endpoints).
- **Headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

---

## Implementation Notes

- **IDs:** Use UUIDv4 for all primary keys.
- **Timestamps:** Store in UTC, return as ISO 8601 with `Z` suffix.
- **Soft deletes:** Optional — consider adding a `deleted_at` field instead of hard deletes.
- **Validation:** Enforce on the server side regardless of client-side checks.
- **CORS:** Allow configured frontend origins; restrict to same-origin in production.
- **Versioning:** Path-based (`/api/v1/`); bump version for breaking changes.
