import { useState } from "react";

export type ClarificationQuestion = {
  qid: string;
  question: string;
  choices?: string[] | null;
  multi_select?: boolean;
};

export type ClarificationRequest = {
  request_id: string;
  question?: string;
  choices?: string[] | null;
  multi_select?: boolean;
  questions?: ClarificationQuestion[];
  answers?: Record<string, string>;
}

type ClarificationCardProps = {
  request: ClarificationRequest;
  onRespond: (answer: string, questionId?: string) => Promise<void>;
};

function ChoiceButtons({
  choices,
  prefix,
  onChoice,
  disabled,
}: {
  choices: string[];
  prefix?: string;
  onChoice: (choice: string) => void;
  disabled: boolean;
}) {
  return <div className="my-2 flex flex-wrap gap-2" aria-label="Clarification choices">
    {choices.map((choice) => <button key={choice} type="button" data-choice={prefix ? `${prefix}:${choice}` : choice} className="rounded border px-2 py-1" disabled={disabled} onClick={() => onChoice(choice)}>{choice}</button>)}
  </div>;
}

export function ClarificationCard({ request, onRespond }: ClarificationCardProps) {
  const batch = request.questions?.filter((question) => question && typeof question.qid === "string") ?? [];
  const [answers, setAnswers] = useState<Record<string, string>>(() => ({ ...request.answers }));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const singleKey = "__single__";

  const setAnswer = (key: string, answer: string) => setAnswers((current) => ({ ...current, [key]: answer }));
  const respond = async (answer: string, questionId?: string) => {
    if (!answer.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      if (questionId) await onRespond(answer, questionId);
      else await onRespond(answer);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };
  const submitBatch = async (event: React.FormEvent) => {
    event.preventDefault();
    for (const question of batch) {
      const answer = answers[question.qid]?.trim() ?? "";
      if (!answer) return;
    }
    setSubmitting(true);
    setError(null);
    try {
      for (const question of batch) await onRespond(answers[question.qid].trim(), question.qid);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div role="dialog" aria-label="Clarification required" aria-busy={submitting} className="rounded-md border border-blue-500/40 bg-blue-500/10 p-3 text-sm">
      <strong>Clarification required</strong>
      {error && <div role="alert" className="mt-2 text-destructive">{error}</div>}
      {batch.length > 0 ? <form onSubmit={submitBatch}>
        <div className="mt-2 space-y-3">
          {batch.map((question) => <fieldset key={question.qid} className="space-y-1">
            <legend className="font-medium">{question.question}</legend>
            {question.choices?.length ? <ChoiceButtons choices={question.choices} prefix={question.qid} disabled={submitting} onChoice={(choice) => setAnswer(question.qid, choice)} /> : null}
            <label className="sr-only" htmlFor={`clarification-${question.qid}`}>{question.question}</label>
            <input id={`clarification-${question.qid}`} aria-label={question.question} className="w-full rounded border bg-background px-2 py-1" value={answers[question.qid] ?? ""} onChange={(event) => setAnswer(question.qid, event.target.value)} disabled={submitting} />
          </fieldset>)}
        </div>
        <button type="submit" className="mt-2 rounded bg-primary px-3 py-1 text-primary-foreground" disabled={submitting}>{submitting ? "Submitting…" : "Submit"}</button>
      </form> : <>
        <p className="mt-1 whitespace-pre-wrap break-words">{request.question || "Please provide an answer."}</p>
        {request.choices?.length ? <ChoiceButtons choices={request.choices} disabled={submitting} onChoice={(choice) => void respond(choice)} /> : null}
        <div className="flex gap-2">
          <label className="sr-only" htmlFor="clarification-answer">Clarification answer</label>
          <input id="clarification-answer" aria-label="Clarification answer" className="min-w-0 flex-1 rounded border bg-background px-2 py-1" value={answers[singleKey] ?? ""} onChange={(event) => setAnswer(singleKey, event.target.value)} disabled={submitting} />
          <button type="button" className="rounded bg-primary px-3 py-1 text-primary-foreground" onClick={() => void respond(answers[singleKey] ?? "")} disabled={submitting}>{submitting ? "Submitting…" : "Submit"}</button>
        </div>
      </>}
    </div>
  );
}
