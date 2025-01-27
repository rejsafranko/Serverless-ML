INSERT INTO table_name (
    Patient_Number, Sadness, Euphoric, Exhausted, Sleep_dissorder,
    Mood_Swing, Suicidal_thoughts, Anorxia, Authority_Respect,
    Try_Explanation, Aggressive_Response, Ignore_Move_On,
    Nervous_Break_down, Admit_Mistakes, Overthinking,
    Sexual_Activity, Concentration, Optimisim, Expert_Diagnose
)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    Sadness = VALUES(Sadness),
    Euphoric = VALUES(Euphoric),
    Exhausted = VALUES(Exhausted),
    Sleep_dissorder = VALUES(Sleep_dissorder),
    Mood_Swing = VALUES(Mood_Swing),
    Suicidal_thoughts = VALUES(Suicidal_thoughts),
    Anorxia = VALUES(Anorxia),
    Authority_Respect = VALUES(Authority_Respect),
    Try_Explanation = VALUES(Try_Explanation),
    Aggressive_Response = VALUES(Aggressive_Response),
    Ignore_Move_On = VALUES(Ignore_Move_On),
    Nervous_Break_down = VALUES(Nervous_Break_down),
    Admit_Mistakes = VALUES(Admit_Mistakes),
    Overthinking = VALUES(Overthinking),
    Sexual_Activity = VALUES(Sexual_Activity),
    Concentration = VALUES(Concentration),
    Optimisim = VALUES(Optimisim),
    Expert_Diagnose = VALUES(Expert_Diagnose);
