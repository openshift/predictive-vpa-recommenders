FROM python:3.8-slim
ADD . /predictive-vpa-recommenders/
RUN cd /predictive-vpa-recommenders && pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
CMD [ "python", "/predictive-vpa-recommenders/main.py"]
