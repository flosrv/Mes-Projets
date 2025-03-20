from retry_requests import retry
import requests, pandas as pd
from datetime import datetime, timedelta, timezone, time
import os,json, re, sys, psycopg2, sys, subprocess, random, warnings, folium, requests_cache, openmeteo_requests, motor
from ndbc_api import NdbcApi
api = NdbcApi()
from pathlib import Path
from datetime import datetime
from requests.exceptions import HTTPError  # Importer HTTPError
from urllib.error import HTTPError
from siphon.simplewebservice.ndbc import NDBC
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, TIMESTAMP, text, String
from sqlalchemy.exc import ProgrammingError
from urllib.parse import quote_plus
from IPython.core.display import *
from sqlalchemy.exc import NoSuchTableError
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pymongo.mongo_client import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import asyncio
from pandas import json_normalize

