"""Authorization for one selected vault fill document, separate from routing."""
from typing import Optional


def fill_origin_refusal(kind: str, page_origin: str, fill_origin: str,
                        frame_route: Optional[dict[str, str]]) -> Optional[dict[str, object]]:
    """Return a refusal before inspection/resolution, or approve this exact pair."""
    if kind != 'login' and (frame_route is not None or fill_origin != page_origin):
        return {
            'success': False,
            'error_type': 'cross_origin_non_login_refused',
            'error': 'Refused: payment and address fills are limited to the bound top-level page.',
        }
    if kind == 'login' and fill_origin != page_origin:
        if frame_route is None:
            return {'success': False, 'error_type': 'cross_origin_route_invalid',
                    'error': 'Could not bind the selected cross-origin frame to its current document.'}
        from tools.approval_prompt import request_elicitation_consent

        decision = request_elicitation_consent(
            f'Fill login from {page_origin} into embedded frame {fill_origin}',
            'This one-time action sends the selected vault login password from the saved top-level site '
            'to this exact embedded origin. Approve only if you recognize both normalized origins.',
            surface='vault-cross-origin-login', title='Confirm cross-origin login fill?',
        )
        if decision != 'accept':
            return {'success': False, 'error_type': 'cross_origin_declined',
                    'error': 'Cross-origin login fill was not approved. Nothing was resolved or written.'}
    return None
