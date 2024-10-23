-- Stored procedure for triggering distribution drift detection.
-- Distribution drift detection is triggered when 100 new data rows are stored.

CREATE PROCEDURE update_row_counter_and_trigger_train_lambda()
BEGIN
    DECLARE current_row_count INT;

    SELECT counter INTO current_row_count
    FROM searches.row_counter;

    SET current_row_count = current_row_count + 1;

    UPDATE mental_health_assesment.row_counter
    SET counter = current_row_count;

    IF current_row_count = 100 THEN
        UPDATE searches.row_counter
        SET counter = 0;

        CALL mysql.lambda_async(
            "arn",
            "JSON_PAYLOAD"
        );
    END IF;

END;
