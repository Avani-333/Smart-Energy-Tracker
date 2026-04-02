"""Fallback Vercel Python entrypoint at repository root."""


def app(environ, start_response):
    body = (
        "Smart Energy Tracker is a Streamlit app. "
        "Use Streamlit Community Cloud or Render for full dashboard hosting."
    ).encode("utf-8")

    status = "200 OK"
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]

    start_response(status, headers)
    return [body]
