
FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/



WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked


COPY predict.py xgboost_model.bin dv.bin requirements.txt ./ 

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "3000"]