-- Create "verified_searches" table.
CREATE TABLE
    IF NOT EXISTS searches.verified_searches (query VARCHAR(255), completion VARCHAR(255))
    -- Create "unverified_searches" table.
CREATE TABLE
    IF NOT EXISTS searches.unverified_searches (query VARCHAR(255), completion VARCHAR(255))