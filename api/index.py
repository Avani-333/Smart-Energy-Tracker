"""Vercel Python serverless entrypoint.

This repository's main UI is a Streamlit app, which is not suited to Vercel's
serverless runtime. This endpoint provides a deployable landing response so
Vercel builds succeed.
"""


def app(environ, start_response):
    body = (
        "Smart Energy Tracker is a Streamlit application. "
        "Deploy the dashboard on Streamlit Community Cloud or Render."
    ).encode("utf-8")

    status = "200 OK"
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]

    start_response(status, headers)
    return [body]
