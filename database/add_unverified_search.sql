-- Ingest user input and model prediction as new data row.
INSERT INTO searches.unverified_searches (query, completion) VALUES (%s, %s)