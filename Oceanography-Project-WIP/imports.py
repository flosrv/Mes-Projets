import retry_requests
from retry_requests import retry
import requests, pandas as pd
from datetime import datetime, timedelta, timezone, time
import os,json, re, sys, psycopg2, sys, subprocess, random, warnings
import folium, requests_cache, openmeteo_requests, motor
from ndbc_api import NdbcApi
api = NdbcApi()
from pathlib import Path
from datetime import datetime
from requests.exceptions import HTTPError  # Importer HTTPError
from urllib.error import HTTPError
from siphon.simplewebservice.ndbc import NDBC
from sqlalchemy import create_engine, MetaData, Table, Column, inspect, Integer, Float, TIMESTAMP, text, String
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.dialects.postgresql import insert


from urllib.parse import quote_plus, unquote
from IPython.core.display import *
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pymongo.mongo_client import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from pandas import json_normalize
import numpy as np
from folium import CircleMarker
import geopandas as gpd
from shapely.geometry import Point
from IPython.core.display import *
