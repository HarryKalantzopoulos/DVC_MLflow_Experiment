FROM python:3.9-slim

WORKDIR /demo

# Copy necessary folders
COPY ./code ./code
COPY ./.dvc/config ./.dvc/config
COPY ./.git ./.git
# Copy necessary files
COPY ./data.dvc ./params.yaml ./requirements.txt ./dvc.lock ./dvc.yaml ./
RUN pip install --no-cache-dir -r ./requirements.txt

RUN dvc pull  && apt-get update && apt-get install git -y
CMD ["sh","-c", "dvc repro -f && python code/register_model.py"]