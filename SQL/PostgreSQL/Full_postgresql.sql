-- ====================================================================
-- 1. Création des Schémas pour l'Architecture Medallion (Bronze, Silver, Gold)
-- ====================================================================

-- Schéma pour les données brutes (Bronze) : données non nettoyées, semi-structurées ou non-structurées
CREATE SCHEMA IF NOT EXISTS data_warehouse.bronze;

-- Schéma pour les données transformées (Silver) : données nettoyées et agrégées
CREATE SCHEMA IF NOT EXISTS data_warehouse.silver;

-- Schéma pour les données prêtes à l'analyse (Gold) : données préparées et agrégées pour BI
CREATE SCHEMA IF NOT EXISTS data_warehouse.gold;

-- ====================================================================
-- 2. Création des Tables pour la Phase Bronze (Données Brutes)
-- ====================================================================

-- Table des données brutes des capteurs (raw_sensor_data) : données semi-structurées sous format JSONB
CREATE TABLE IF NOT EXISTS data_warehouse.bronze.raw_sensor_data (
    raw_data_id SERIAL PRIMARY KEY,               -- Identifiant unique
    sensor_id INT NOT NULL,                      -- Identifiant du capteur
    raw_json JSONB NOT NULL,                     -- Données brutes au format JSON
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Date et heure de réception des données
);

-- ====================================================================
-- 3. Création des Tables pour la Phase Silver (Données Transformées)
-- ====================================================================

-- Table des capteurs nettoyés et enrichis (silver_sensors) : informations transformées et enrichies des capteurs
CREATE TABLE IF NOT EXISTS data_warehouse.silver.silver_sensors (
    sensor_id SERIAL PRIMARY KEY, 
    location VARCHAR(255),                         -- Localisation du capteur
    model VARCHAR(100),                            -- Modèle du capteur
    manufacturer VARCHAR(100),                     -- Fabricant du capteur
    status VARCHAR(20),                            -- Statut du capteur ('active', 'inactive')
    installed_at TIMESTAMP,                        -- Date d'installation
    last_maintenance TIMESTAMP,                    -- Dernière maintenance
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date de nettoyage des données
);

-- Table des lectures nettoyées et agrégées des capteurs (silver_sensor_readings) : transformation des données brutes en lecture propre
CREATE TABLE IF NOT EXISTS data_warehouse.silver.silver_sensor_readings (
    reading_id SERIAL PRIMARY KEY,
    sensor_id INT REFERENCES data_warehouse.silver.silver_sensors(sensor_id), -- Référence au capteur
    timestamp TIMESTAMP NOT NULL,                  -- Heure de la lecture
    value DOUBLE PRECISION NOT NULL,               -- Valeur mesurée par le capteur
    unit VARCHAR(50),                              -- Unité de mesure (Celsius, PSI, etc.)
    status VARCHAR(20) CHECK (status IN ('normal', 'warning', 'error')),  -- Statut de la lecture
    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Date de nettoyage
);

-- ====================================================================
-- 4. Création des Tables pour la Phase Gold (Données Prêtes à l'Analyse)
-- ====================================================================

-- Table des mesures agrégées des capteurs (gold_sensor_metrics) : données prêtes à l'analyse pour BI
CREATE TABLE IF NOT EXISTS data_warehouse.gold.gold_sensor_metrics (
    sensor_id INT PRIMARY KEY,
    avg_value DOUBLE PRECISION,                     -- Moyenne des valeurs de lecture
    min_value DOUBLE PRECISION,                     -- Valeur minimale
    max_value DOUBLE PRECISION,                     -- Valeur maximale
    total_records BIGINT,                           -- Nombre total de lectures
    period_start TIMESTAMP,                         -- Début de la période
    period_end TIMESTAMP,                           -- Fin de la période
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date de traitement des données
);

-- ====================================================================
-- 5. Vues pour l'Accès à l'Analyse BI (Gold Phase) avec Accès Restreint
-- ====================================================================

-- Vue pour accéder aux données agrégées (Gold) des capteurs prêtes à l'analyse pour les utilisateurs BI
CREATE VIEW data_warehouse.gold.vw_sensor_metrics AS
SELECT 
    sensor_id,
    avg_value,
    min_value,
    max_value,
    total_records,
    period_start,
    period_end
FROM data_warehouse.gold.gold_sensor_metrics;

-- ====================================================================
-- 6. Pipeline de Transformation des Données : de Bronze à Silver à Gold
-- ====================================================================

-- Etape 1: Chargement des données dans la phase Bronze (Données Brutes)
INSERT INTO data_warehouse.bronze.raw_sensor_data (sensor_id, raw_json)
SELECT sensor_id, raw_json FROM raw_data_source;  -- Importe les données brutes (raw_data_source est une source fictive ici)

-- Etape 2: Transformation des données de la phase Bronze (Brute) vers la phase Silver (Transformée et Nettoyée)
INSERT INTO data_warehouse.silver.silver_sensors (sensor_id, location, model, manufacturer, status, installed_at, last_maintenance)
SELECT 
    sensor_id,
    raw_json->>'location' AS location,
    raw_json->>'model' AS model,
    raw_json->>'manufacturer' AS manufacturer,
    raw_json->>'status' AS status,
    (raw_json->>'installed_at')::timestamp AS installed_at,
    (raw_json->>'last_maintenance')::timestamp AS last_maintenance
FROM data_warehouse.bronze.raw_sensor_data;

INSERT INTO data_warehouse.silver.silver_sensor_readings (sensor_id, timestamp, value, unit, status)
SELECT 
    sensor_id,
    (raw_json->>'timestamp')::timestamp AS timestamp,
    (raw_json->>'value')::double precision AS value,
    raw_json->>'unit' AS unit,
    raw_json->>'status' AS status
FROM data_warehouse.bronze.raw_sensor_data
WHERE raw_json->>'status' IN ('normal', 'warning', 'error');

-- Etape 3: Agrégation des données transformées de la phase Silver vers la phase Gold
INSERT INTO data_warehouse.gold.gold_sensor_metrics (sensor_id, avg_value, min_value, max_value, total_records, period_start, period_end)
SELECT 
    sensor_id,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    COUNT(*) AS total_records,
    MIN(timestamp) AS period_start,
    MAX(timestamp) AS period_end
FROM data_warehouse.silver.silver_sensor_readings
GROUP BY sensor_id;

-- ====================================================================
-- 7. Index et Optimisation pour le Traitement des Données Big Data
-- ====================================================================

-- Création d'index pour accélérer les requêtes sur les grandes tables
CREATE INDEX IF NOT EXISTS idx_bronze_raw_sensor_data_received_at ON data_warehouse.bronze.raw_sensor_data (received_at);
CREATE INDEX IF NOT EXISTS idx_silver_sensor_readings_timestamp ON data_warehouse.silver.silver_sensor_readings (timestamp);

-- ====================================================================
-- 8. Sécurité et Gestion des Accès aux Données
-- ====================================================================

-- Création de rôles pour l'accès contrôlé aux différentes phases
CREATE ROLE IF NOT EXISTS data_scientist;  -- Rôle pour les Data Scientists
CREATE ROLE IF NOT EXISTS bi_user;          -- Rôle pour les utilisateurs BI

-- Restriction de l'accès direct aux tables (tables en Bronze et Silver) pour éviter la manipulation directe des données
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA data_warehouse.bronze FROM bi_user;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA data_warehouse.silver FROM bi_user;

-- Attribution d'accès uniquement aux vues dans la phase Gold pour BI
GRANT SELECT ON ALL VIEWS IN SCHEMA data_warehouse.gold TO bi_user;
GRANT SELECT ON ALL VIEWS IN SCHEMA data_warehouse.gold TO data_scientist;

-- ====================================================================
-- 9. Nettoyage et Archivage des Données (Politique de Rétention)
-- ====================================================================

-- Suppression des données brutes de plus de 6 mois dans la phase Bronze
DELETE FROM data_warehouse.bronze.raw_sensor_data WHERE received_at < CURRENT_DATE - INTERVAL '6 months';

-- Suppression des lectures de capteurs de plus de 1 an dans la phase Silver
DELETE FROM data_warehouse.silver.silver_sensor_readings WHERE timestamp < CURRENT_DATE - INTERVAL '1 year';

-- ====================================================================
-- 10. Monitoring et Logs pour les Opérations Data Engineering
-- ====================================================================

-- Table pour enregistrer les logs des opérations de transformation des données
CREATE TABLE IF NOT EXISTS monitoring.data_transformation_logs (
    log_id SERIAL PRIMARY KEY,              -- Identifiant du log
    operation VARCHAR(255),                 -- Description de l'opération
    status VARCHAR(20),                     -- Statut de l'opération (success, fail)
    error_message TEXT,                     -- Message d'erreur en cas d'échec
    log_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Date et heure du log
);

-- Enregistrement d'une opération réussie dans les logs
INSERT INTO monitoring.data_transformation_logs (operation, status)
VALUES ('Data Transformation from Bronze to Silver', 'success');

-- ====================================================================
-- FIN DU SCRIPT : Architecture Medallion (Bronze, Silver, Gold)
-- ====================================================================