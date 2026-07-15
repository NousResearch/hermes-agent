export type ApprovalRisk = {
  riskLabel?: { en: string; zh: string }
  riskWarning?: { en: string; zh: string }
}

export function approvalRiskText(risk: ApprovalRisk): string | null {
  if (
    !risk.riskLabel?.en ||
    !risk.riskLabel.zh ||
    !risk.riskWarning?.en ||
    !risk.riskWarning.zh
  ) {
    return null
  }

  return `${risk.riskLabel.en} / ${risk.riskLabel.zh} — ${risk.riskWarning.en} / ${risk.riskWarning.zh}`
}
