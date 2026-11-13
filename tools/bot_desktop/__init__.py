"""Bot Desktop: per-profile headless Xfce desktop over RFB, viewed from Hermes Desktop.

``runtime`` owns the Xvnc/Xfce process and the published env; ``lease`` owns who may drive the
screen (agent vs. human); ``rfb_filter`` is the byte-level input gate the WebSocket bridge applies.
"""
