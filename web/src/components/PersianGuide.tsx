import { useState } from "react";
import { ChevronDown, BookOpen } from "lucide-react";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * راهنمای فارسی — a Persian quick-start guide shown on the Documentation page
 * while the active locale is Persian (or Arabic, which shares the RTL
 * treatment). The upstream docs site is English-only, so this gives Persian
 * users a localized overview of every dashboard section plus the key CLI
 * commands. Collapsible so it never blocks access to the real docs iframe.
 */

interface GuideSection {
  title: string;
  body: string;
}

const GUIDE_SECTIONS: GuideSection[] = [
  {
    title: "💬 گفتگو",
    body: "با عامل خود گفتگو کنید. پیام تایپ کنید، Enter بزنید و پاسخ را ببینید. می‌توانید فایل پیوست کنید، مدل را عوض کنید و از دستورهای اسلش (مثل /help) استفاده کنید.",
  },
  {
    title: "🗂 نشست‌ها",
    body: "تاریخچه همه گفتگوها اینجاست. با فیلترهای «گفتگوها / خودکارسازی / همه» جستجو کنید، نشست‌های قدیمی را حذف کنید یا خروجی JSON نشست‌ها را دوباره وارد کنید.",
  },
  {
    title: "📊 تحلیل‌ها و مدل‌ها",
    body: "مصرف توکن، فراخوانی‌های API، هزینه تخمینی و تفکیک روزانه بر اساس مدل را ببینید تا مصرف خود را مدیریت کنید.",
  },
  {
    title: "🕐 زمان‌بندی (Cron)",
    body: "وظایف تکرارشونده بسازید: هر روز ساعت ۹ صبح خلاصه خبر بده، هر هفته گزارش ایمیل کن و... خروجی را به کانال دلخواه (تلگرام، ایمیل و...) تحویل بدهید.",
  },
  {
    title: "🧩 مهارت‌ها و افزونه‌ها",
    body: "مهارت‌ها قابلیت‌های آماده‌ای هستند که عامل از ~/.hermes/skills/ بارگذاری می‌کند. افزونه‌ها را از GitHub نصب کنید یا ارائه‌دهنده حافظه را عوض کنید.",
  },
  {
    title: "🔗 کانال‌ها و اتصال‌ها",
    body: "در «کانال‌ها» پلتفرم‌های پیام‌رسانی (تلگرام، Discord، Slack، واتساپ) را وصل کنید. «وب‌هوک‌ها» برای اتصال سرویس‌های خارجی و «جفت‌سازی» برای اتصال امن دستگاه‌های جدید است.",
  },
  {
    title: "🔑 کلیدها و پیکربندی",
    body: "کلیدهای API ارائه‌دهندگان (OpenAI، Anthropic و...) را در «کلیدها» وارد کنید — تغییرات بلافاصله ذخیره می‌شوند. تنظیمات پیشرفته در «پیکربندی» (فایل config.yaml) است.",
  },
  {
    title: "🖥 سیستم",
    body: "وضعیت دروازه، راه‌اندازی مجدد، به‌روزرسانی Hermes و گزارش‌های لحظه‌ای از اینجا در دسترس است. اگر چیزی کار نکرد، اول «گزارش‌ها» را ببینید.",
  },
];

/**
 * راهنمای شروع سریع — the guide.html first-run walkthrough as data:
 * five figures, Persian captions identical to guide.html's <figcaption>s
 * (numbered ۱–۵). Images are the freshly recaptured guide screenshots,
 * served from public/guide-images/ (present in dev and the prod build).
 * Keep caption text in sync with guide.html — same section, same order.
 */
const WALKTHROUGH: Array<{ src: string; caption: string }> = [
  {
    src: "guide-images/01-first-launch.png",
    caption: "۱ — برنامه باز شد؛ در اولین اجرا زبان پیش‌فرض انگلیسی است.",
  },
  {
    src: "guide-images/02-language-menu.png",
    caption: "۲ — در پایین نوار کناری روی «Switch language» کلیک کنید و فارسی را انتخاب کنید.",
  },
  {
    src: "guide-images/03-persian-rtl.png",
    caption: "۳ — کل رابط بلافاصله فارسی و راست‌به‌چپ می‌شود؛ نوار کناری به سمت راست می‌رود.",
  },
  {
    src: "guide-images/04-models-persian.png",
    caption: "۴ — از بخش «مدل‌ها» ارائه‌دهنده (Nous Portal، OpenRouter، OpenAI و…) و مدل دلخواه را تنظیم کنید.",
  },
  {
    src: "guide-images/05-guide-persian.png",
    caption: "۵ — همین راهنما همیشه در دسترس است: بخش «مستندات» در خود برنامه.",
  },
];

const CLI_TIPS: Array<{ cmd: string; desc: string }> = [
  { cmd: "hermes", desc: "شروع گفتگوی تعاملی در ترمینال" },
  { cmd: "hermes gateway", desc: "اجرای دروازه (کانال‌ها و API سرور)" },
  { cmd: "hermes status", desc: "مشاهده وضعیت سرویس‌ها" },
  { cmd: "hermes update", desc: "به‌روزرسانی Hermes" },
];

/**
 * راه‌اندازی سریع در ۶ گام — mirror of the guide.html بخش ۱۷ (id=quickstart).
 * Commands must stay verbatim-identical to guide.html; keep row-for-row in sync.
 */
const QUICKSTART: Array<{ cmd: string; desc: string }> = [
  {
    cmd: "npm install",
    desc: "از ریشهٔ مخزن (تک‌قفل‌نامه: هرگز از زیرپوشه‌ها مثل ui-tui نصب نکنید). بک‌اند: pip install -e \".[web,dev]\". آزمون سلامت: hermes doctor.",
  },
  {
    cmd: "hermes setup --portal",
    desc: "وارد حساب Nous Portal شوید، مدل را انتخاب کنید و دروازه ابزارها (وب، تصویر، TTS، مرورگر) همان‌جا وصل می‌شود. انتخاب تک‌تک ابزارها: hermes setup tools.",
  },
  {
    cmd: "hermes gateway run",
    desc: "اجرای دروازه. بازرسی: hermes gateway status باید «در حال اجرا» را نشان دهد.",
  },
  {
    cmd: "hermes dashboard --port 9119",
    desc: "پس از چند ده ثانیه http://127.0.0.1:9119/ باز می‌شود (پرچم --no-open هم هست).",
  },
  {
    cmd: "hermes config set display.language fa",
    desc: "انتخاب زبان فارسی؛ در داشبورد هم دکمهٔ «تغییر زبان» در پایین نوار کناری. فونت Vazirmatn و چیدمان راست‌به‌چپ خودکار فعال می‌شود.",
  },
  {
    cmd: "start-hermes-stack.cmd",
    desc: "بررسی نهایی: hermes doctor بدون هشدار وابستگی + «وضعیت دروازه: در حال اجرا» در داشبورد. برای شروع خودکار پس از هر ری‌استارت ویندوز، این اسکریپت را یک‌بار اجرا کنید.",
  },
];

/**
 * وابستگی‌های اختیاری تست‌ها — mirror of the guide.html بخش ۱۸ (id=test-extras).
 * Extra names must stay identical to pyproject [project.optional-dependencies].
 */
const TEST_EXTRAS: Array<{ extra: string; cmd: string; desc: string }> = [
  {
    extra: "dev",
    cmd: "pip install -e \".[dev]\"",
    desc: "خود pytest و pytest-asyncio، mcp، httpx، ruff و ty — پیش‌نیاز هر اجرای تستی",
  },
  {
    extra: "messaging",
    cmd: "pip install -e \".[messaging]\"",
    desc: "تست‌های تلگرام، Discord، Slack و کانال‌های پیام‌رسانی (tests/gateway، tests/tools)",
  },
  {
    extra: "anthropic",
    cmd: "pip install -e \".[anthropic]\"",
    desc: "تست‌های آداپتور Anthropic در tests/agent (سایر ارائه‌دهنده‌ها از طریق کلید API قلابی stub می‌شوند)",
  },
  {
    extra: "web",
    cmd: "pip install -e \".[web]\"",
    desc: "سرور داشبورد/دروازه که تست‌های یکپارچگی به آن وصل می‌شوند",
  },
  {
    extra: "pytest-timeout · pytest-xdist",
    cmd: "pip install pytest-timeout pytest-xdist",
    desc: "اجرای قطعه‌قطعهٔ suite دروازه روی ویندوز با تایم‌اوت در هر تست (فقط برای توسعه؛ پین نشده‌اند)",
  },
];

/**
 * عیب‌یابی — full mirror of the guide.html بخش ۱۷ troubleshooting table:
 * all 15 rows in table order (IPC bridge, desktop setup, (-6) dist error,
 * WS "session token" boot failure, pip drift, npm drift, postinstall
 * scripts, first run, gateway, backend version, terminal font, language
 * apply, hermes doctor, credit, MCP). Keep row-for-row in sync.
 */
const TROUBLESHOOTING: Array<{ problem: string; fix: string }> = [
  {
    problem: "«پل IPC دسکتاپ در دسترس نیست» (Desktop IPC bridge is unavailable)",
    fix: "یعنی نشانی سرور توسعه (127.0.0.1:5174) را در یک مرورگر معمولی باز کرده‌اید — پل IPC فقط در پنجره الکترون تزریق می‌شود. پنجره برنامه دسکتاپ Hermes را باز کنید؛ صفحه در مرورگر بالا می‌آید اما برنامه دسکتاپ نیست.",
  },
  {
    problem: "راه‌اندازی دسکتاپ ناموفق بود",
    fix: "از صفحه خطا «تلاش دوباره»، «تعمیر نصب» یا «استفاده از دروازه محلی» را امتحان کنید؛ هیچ‌کدام گفتگوها یا تنظیمات را حذف نمی‌کنند. لاگ‌ها از دکمه «باز کردن لاگ‌ها» در دسترس‌اند.",
  },
  {
    problem: "«Hermes couldn't start the desktop UI» با کد (-6) و مسیر apps\\desktop\\dist\\index.html",
    fix: "برنامه بدون متغیر محیطیِ سرور توسعه اجرا شده: الکترون به‌دنبال باندل ساخته‌شده (dist) می‌گردد که در مخزن توسعه وجود نیست (-6 = فایل پیدا نشد). برنامه را با cd apps/desktop و سپس npm run dev اجرا کنید؛ اجرای مستقیم electron . دقیقاً همین خطا را می‌دهد. گزینه hermes desktop --force-build فقط برای نسخه نصب‌شده کاربرد دارد.",
  },
  {
    problem: "«Backend exited before it became ready» همراه با «rejected the session token»",
    fix: "پیامِ توکن گمراه‌کننده است — علت واقعی تقریباً همیشه ناهم‌خوانی وابستگی‌های پایتون است (uvicorn قدیمی، websockets ناسازگار یا نبودن concurrent_log_handler). در desktop.log دنبال Traceback واقعی بگردید و روی همان مفسر باک‌اند pip install -e \".[web]\" (یا uv sync --extra web) را اجرا کنید.",
  },
  {
    problem: "چک‌لیست ناهم‌خوانی محیط پایتون (خطاهای import یا نسخه)",
    fix: "۱) pip check باید «No broken requirements found» بدهد. ۲) نسخه‌ها را با پین‌های pyproject.toml بسنجید و با pip install -e \".[web]\" هم‌تراز کنید. ۳) اگر pip هشدار «Ignoring invalid distribution ~xyz» داد، پوشه‌های «~...» را در Lib/site-packages حذف کنید (خرده‌نصب خراب). ۴) آزمون نهایی: python -c \"import uvicorn, fastapi, websockets; from tui_gateway.ws import handle_ws\" — بدون خطا یعنی محیط سالم است.",
  },
  {
    problem: "چک‌لیست ناهم‌خوانی محیط npm/Node (خطاهای نصب، EENGINE)",
    fix: "۱) نصب‌های تکراری را با npm ci انجام دهید تا node_modules دقیقاً مطابق package-lock.json بازسازی شود؛ npm install ممکن است lockfile را جابه‌جا کند. ۲) بهداشت lockfile: این مخزن تک‌قفل‌نامه است — نصب از ریشه مخزن انجام شود و package-lock.json همراه هر تغییر وابستگی کامیت گردد؛ npm install در زیرپوشه‌ها (مثل ui-tui) رزولوشن پکیج‌های داخلی را می‌شکند. ۳) خطای EENGINE یعنی npm نصب‌شده در بازه مجاز engines پروژه نیست (<11.10.0 || >=11.17.0) — با npm i -g npm@12 هم‌تراز کنید و از دورزدن با --engine-strict=false جز در موارد ضروری پرهیز کنید.",
  },
  {
    problem: "خطای EALLOWSCRIPTS یا اجرا نشدن اسکریپت‌های postinstall",
    fix: "این خطا یعنی postinstallهای بسته‌ها (مثل دانلود باینری Electron) اجرا نشده‌اند. علت رایج: خط allow-scripts در ~/.npmrc کاربر — آن خط را حذف کنید تا پیکربندی پروژه حاکم شود. اگر ممنوعیت اسکریپت عمدی است، به‌جای غیرفعال‌سازی سراسری از npm ci --ignore-scripts آگاهانه استفاده کنید. آزمون سلامت: باینری Electron باید در node_modules/electron/dist باشد؛ نبودش یعنی postinstall اجرا نشده است.",
  },
  {
    problem: "اولین اجرا طول می‌کشد",
    fix: "در اولین اجرا، برنامه باک‌اند را از صفر راه می‌اندازد (نصب وابستگی‌ها)؛ صفحه پیشرفت را باز بگذارید. اجراهای بعدی سریع‌اند.",
  },
  {
    problem: "اتصال با دروازه قطع شد",
    fix: "برنامه خودش در پس‌زمینه تلاش مجدد می‌کند؛ اگر باقی ماند، تنظیمات دروازه را باز کنید و برای دروازه راه دور دوباره وارد شوید.",
  },
  {
    problem: "«باک‌اند قدیمی است»",
    fix: "زمان اجرای Hermes از برنامه دسکتاپ عقب است — از اعلان «به‌روزرسانی Hermes» استفاده کنید.",
  },
  {
    problem: "متن فارسی به‌هم‌ریخته در ترمینال",
    fix: "فونت ترمینال را در تنظیمات ← ظاهر به یک فونت پشتیبان RTL (مثل Vazirmatn یا JetBrains Mono) تغییر دهید.",
  },
  {
    problem: "فارسی اعمال نشد",
    fix: "در ترمینال: hermes config get display.language — مقدار باید fa باشد. متغیر HERMES_LANGUAGE بر پیکربندی اولویت دارد.",
  },
  {
    problem: "خطای کلی",
    fix: "از ترمینال hermes doctor را اجرا کنید؛ برای ارسال گزارش به Nous از «ارسال عیب‌یابی» استفاده کنید (اسرار سانسور و بسته پس از ۱۴ روز حذف می‌شود).",
  },
  {
    problem: "اعتبار تمام شد",
    fix: "از هشدار «افزودن اعتبار» یا بخش صورت‌حساب در Portal ادامه دهید؛ مدل‌های جایگزین در تنظیمات مدل توقف‌گاه میانی می‌سازند.",
  },
  {
    problem: "سرویس MCP در دسترس نیست",
    fix: "از تنظیمات ← MCP حالت سرور را ببینید؛ اگر «نیازمند احراز هویت» است دوباره وارد شوید و «بارگذاری مجدد MCP» را بزنید.",
  },
];

export function PersianGuide() {
  const { locale } = useI18n();
  const [open, setOpen] = useState(true);

  // Only shown for RTL locales — the English docs iframe covers everyone else.
  if (locale !== "fa" && locale !== "ar") return null;

  return (
    <div
      dir="rtl"
      className={cn(
        "mb-3 rounded-sm border border-primary/25 bg-primary/[0.04]",
        "shadow-[inset_0_1px_0_0_#ffffff12]",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className={cn(
          "flex w-full items-center gap-2 px-4 py-3 text-start",
          "hover:bg-primary/[0.06] transition-colors",
        )}
      >
        <BookOpen className="size-4 shrink-0 text-primary" />
        <span className="flex-1 text-sm font-bold text-foreground">
          راهنمای استفاده از داشبورد
        </span>
        <span className="text-xs text-muted-foreground">
          {open ? "بستن" : "نمایش"}
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div className="border-t border-primary/15 px-4 py-3">
          <p className="mb-3 text-xs leading-relaxed text-muted-foreground">
            این داشبورد کنترل‌پنل کامل عامل Hermes است — گفتگو، خودکارسازی،
            اتصال به پیام‌رسان‌ها و مدیریت پیکربندی، همه از مرورگر. بخش‌های
            زیر را به ترتیب بررسی کنید تا سریع شروع کنید:
          </p>

          <ol className="grid gap-2 sm:grid-cols-2">
            {GUIDE_SECTIONS.map((section, i) => (
              <li
                key={section.title}
                className={cn(
                  "rounded-sm border border-current/10 bg-background/40",
                  "px-3 py-2",
                )}
              >
                <div className="mb-0.5 flex items-center gap-1.5">
                  <span className="inline-flex size-4 items-center justify-center rounded-full bg-primary/15 text-[10px] font-bold text-primary tabular-nums">
                    {i + 1}
                  </span>
                  <span className="text-xs font-bold text-foreground">
                    {section.title}
                  </span>
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {section.body}
                </p>
              </li>
            ))}
          </ol>

          <div className="mt-3 rounded-sm border border-current/10 bg-background/40 px-3 py-2">
            <div className="mb-1.5 text-xs font-bold text-foreground">
              🚀 شروع سریع در ۵ گام
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {WALKTHROUGH.map((fig) => (
                <figure
                  key={fig.src}
                  className="overflow-hidden rounded-sm border border-current/10 bg-background/60"
                >
                  <img
                    src={fig.src}
                    alt={fig.caption}
                    loading="lazy"
                    className="block w-full"
                  />
                  <figcaption className="border-t border-current/10 px-2 py-1.5 text-[11px] leading-relaxed text-muted-foreground">
                    {fig.caption}
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>

          <div className="mt-3 rounded-sm border border-current/10 bg-background/40 px-3 py-2">
            <div className="mb-1.5 text-xs font-bold text-foreground">
              ⚡ راه‌اندازی سریع در ۶ گام
            </div>
            <dl className="grid gap-1">
              {QUICKSTART.map((step, i) => (
                <div key={step.cmd} className="flex items-baseline gap-2">
                  <span className="inline-flex size-4 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[10px] font-bold text-primary tabular-nums">
                    {i + 1}
                  </span>
                  <code
                    dir="ltr"
                    className={cn(
                      "shrink-0 rounded-sm bg-muted px-1.5 py-0.5",
                      "font-mono text-[11px] text-primary",
                    )}
                  >
                    {step.cmd}
                  </code>
                  <dd className="text-[11px] leading-relaxed text-muted-foreground">
                    {step.desc}
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="mt-3 rounded-sm border border-current/10 bg-background/40 px-3 py-2">
            <div className="mb-1.5 text-xs font-bold text-foreground">
              🛠 عیب‌یابی رایج
            </div>
            <dl className="grid gap-2">
              {TROUBLESHOOTING.map((item) => (
                <div key={item.problem} className="rounded-sm bg-background/40 px-2 py-1.5">
                  <dt className="text-[11px] font-bold text-foreground">
                    {item.problem}
                  </dt>
                  <dd className="mt-0.5 text-[11px] leading-relaxed text-muted-foreground">
                    {item.fix}
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="mt-3 rounded-sm border border-current/10 bg-background/40 px-3 py-2">
            <div className="mb-1.5 text-xs font-bold text-foreground">
              ⌨️ دستورهای پرکاربرد ترمینال
            </div>
            <dl className="grid gap-1">
              {CLI_TIPS.map((tip) => (
                <div key={tip.cmd} className="flex items-baseline gap-2">
                  <code
                    dir="ltr"
                    className={cn(
                      "shrink-0 rounded-sm bg-muted px-1.5 py-0.5",
                      "font-mono text-[11px] text-primary",
                    )}
                  >
                    {tip.cmd}
                  </code>
                  <dd className="text-[11px] text-muted-foreground">
                    {tip.desc}
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="mt-3 rounded-sm border border-current/10 bg-background/40 px-3 py-2">
            <div className="mb-1.5 text-xs font-bold text-foreground">
              🧪 وابستگی‌های اختیاری تست‌ها (توسعه‌دهندگان)
            </div>
            <dl className="grid gap-1">
              {TEST_EXTRAS.map((item) => (
                <div key={item.extra} className="flex items-baseline gap-2">
                  <code
                    dir="ltr"
                    className={cn(
                      "shrink-0 rounded-sm bg-muted px-1.5 py-0.5",
                      "font-mono text-[11px] text-primary",
                    )}
                  >
                    {item.cmd}
                  </code>
                  <dd className="text-[11px] leading-relaxed text-muted-foreground">
                    <span className="font-bold text-foreground">[{item.extra}]</span>{" "}
                    {item.desc}
                  </dd>
                </div>
              ))}
            </dl>
          </div>

          <p className="mt-3 text-[11px] text-muted-foreground">
            💡 نکته: زبان و پوسته را از دکمه‌های پایین نوار کناری عوض کنید.
            مستندات کامل انگلیسی در قاب زیر است.
          </p>
        </div>
      )}
    </div>
  );
}
