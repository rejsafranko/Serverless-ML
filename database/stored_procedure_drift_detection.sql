-- Stored procedure for drift detection and train lambda triggering. Drift is set to 100 new data rows.
CREATE PROCEDURE update_row_counter_and_trigger_train_lambda () BEGIN DECLARE current_row_count INT;

SELECT
    counter INTO current_row_count
FROM
    searches.row_counter
SET
    current_row_count = current_row_count + 1
UPDATE searches.row_counter
SET
    counter = current_row_count IF current_row_count = 100 THEN
UPDATE searches.row_counter
SET
    counter 0 CALL mysql.lambda_async (
        "arn:aws:lambda:eu-west-1:320329586666:function:MlAutocompleteApiStack-DockerTrain937700CB-kB5UQyk6vdCz",
        "JSON_PAYLOAD"
    ) END IF;

END