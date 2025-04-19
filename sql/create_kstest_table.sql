CREATE TABLE ks_test_results (
    column_name VARCHAR(255) NOT NULL,
    ks_statistic FLOAT NOT NULL,
    p_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (column_name)
);