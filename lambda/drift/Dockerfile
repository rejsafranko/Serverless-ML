FROM public.ecr.aws/lambda/python:3.11-arm64 AS base
ENV PIP_NO_CACHE_DIR=true
WORKDIR /opt

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM public.ecr.aws/lambda/python:3.11-arm64

COPY --from=base /opt /opt

WORKDIR ${LAMBDA_TASK_ROOT}
COPY src/ .

CMD ["drift.handler"]
