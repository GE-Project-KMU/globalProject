"""Gateway TLS: optional project-scoped CA, never disable certificate checks."""
import os
import ssl
from .paths import workspace_path


def gateway_tls(ca_bundle=None):
    path=ca_bundle or os.environ.get("ROAD_CA_BUNDLE")
    if not path:return True
    return ssl.create_default_context(cafile=str(workspace_path(path,exists=True)))
