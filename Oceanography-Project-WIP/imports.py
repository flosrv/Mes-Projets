import openmeteo_requests
import requests_cache
from retry_requests import retry
import subprocess
import sys
import requests, pandas as pd
from datetime import datetime, timedelta, timezone, time
import os,json
import re, json
from ndbc_api import NdbcApi
api = NdbcApi()
from pathlib import Path
from datetime import datetime
from requests.exceptions import HTTPError  # Importer HTTPError
import sys, psycopg2
from urllib.error import HTTPError
from siphon.simplewebservice.ndbc import NDBC
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, TIMESTAMP, text
from sqlalchemy.exc import ProgrammingError
from urllib.parse import quote_plus
import random, folium
from IPython.core.display import *