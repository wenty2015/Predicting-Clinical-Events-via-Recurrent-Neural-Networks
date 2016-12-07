
SET SEARCH_PATH TO MIMICIII;

SELECT COUNT(1) FROM PATIENTS; -- 46520 records
SELECT COUNT(1) FROM ADMISSIONS; -- 58976 records

---------------------------------------------------
 -- patients with more than 1 admissions
 -- 6387 records
 SELECT COUNT(1)
 FROM (SELECT AD.SUBJECT_ID
       FROM ADMISSIONS AD,PATIENTS P
       WHERE AD.SUBJECT_ID = P.SUBJECT_ID
       -- ADMISSION_TYPE:
       --  ‘ELECTIVE’- planned admission
       --  ‘URGENT’,‘EMERGENCY’- unplanned medical care
       --  ‘NEWBORN’- pertains to the patient's birth
       AND ADMISSION_TYPE NOT IN ('NEWBORN')
       AND HOSPITAL_EXPIRE_FLAG = '0'
       -- uncomment if filter the dead patients
       /* AND P.EXPIRE_FLAG = '0'*/
       GROUP BY AD.SUBJECT_ID
       HAVING COUNT(1) > 1)T;

---------------------------------------------------
-- patients with more than 1 admissions
-- each admission is attached with multiple 3-digits ICD9 codes
-- 204160 records
DROP TABLE PATIENT_VISIT;
CREATE TABLE PATIENT_VISIT AS
SELECT DISTINCT AD.SUBJECT_ID, AD.HADM_ID, DATE(AD.ADMITTIME) AS ADMITTIME,
         SUBSTRING(ICD.ICD9_CODE,1,3) AS ICD9_CODE
   FROM (SELECT SUBJECT_ID, HADM_ID, ADMITTIME
          FROM ADMISSIONS AD
          WHERE AD.ADMISSION_TYPE NOT IN ('NEWBORN')
          AND AD.HOSPITAL_EXPIRE_FLAG = '0')AD
   INNER JOIN DIAGNOSES_ICD ICD
    ON (AD.SUBJECT_ID,AD.HADM_ID) = (ICD.SUBJECT_ID,ICD.HADM_ID)
   WHERE
    AD.SUBJECT_ID IN (SELECT AD.SUBJECT_ID
                        FROM PATIENTS P,ADMISSIONS AD
                        WHERE AD.SUBJECT_ID = P.SUBJECT_ID
                        AND AD.ADMISSION_TYPE NOT IN ('NEWBORN')
                        AND AD.HOSPITAL_EXPIRE_FLAG = '0'
                        -- uncomment if filter the dead patients
                        /*AND P.EXPIRE_FLAG = '0'*/
                        GROUP BY AD.SUBJECT_ID
                        HAVING COUNT(1) > 1)
    AND ICD.ICD9_CODE IS NOT NULL
    ORDER BY AD.SUBJECT_ID,AD.HADM_ID, ADMITTIME,ICD9_CODE;

-- check values
-- 0 records
SELECT *
  FROM PATIENT_VISIT
  WHERE SUBJECT_ID IS NULL OR HADM_ID IS NULL
        OR ADMITTIME IS NULL OR ICD9_CODE IS NULL;
-- check if there exists multiple records
-- 0 records
SELECT SUBJECT_ID,HADM_ID,ICD9_CODE
  FROM PATIENT_VISIT
  GROUP BY SUBJECT_ID,HADM_ID,ICD9_CODE
  HAVING COUNT(1) >1;

-- delete patients with only 1 admission
-- 57 records
DELETE FROM PATIENT_VISIT
WHERE SUBJECT_ID IN (SELECT SUBJECT_ID
                      FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
                              FROM PATIENT_VISIT)T
                      GROUP BY SUBJECT_ID
                      HAVING COUNT(1) = 1);
-- statistics:
-- patients with more than 1 admissions
-- 6380 records
SELECT COUNT(DISTINCT SUBJECT_ID)
  FROM PATIENT_VISIT;
-- # of admissions
-- 17130 records
SELECT COUNT(DISTINCT HADM_ID)
  FROM PATIENT_VISIT;

-- patients with # of admissions
-- 27 records
-- more than 5% of patients have [2,3,4] admissions
-- more than 1% of patients have [2,3,4,5,6] admissions
SELECT CNT,COUNT(1)
  FROM
  (SELECT SUBJECT_ID,COUNT(1) AS CNT
    FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
            FROM PATIENT_VISIT)T
    GROUP BY SUBJECT_ID)C
  GROUP BY CNT
  ORDER BY COUNT(1) DESC;

-- # of admissions
-- 17130 records
SELECT COUNT(1)
  FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
         FROM PATIENT_VISIT)T;
---------------------------------------------------
-- get notes for each admissions
-- 474890 records
DROP TABLE PATIENT_VISIT_NOTE;
CREATE TABLE PATIENT_VISIT_NOTE AS
  SELECT N.ROW_ID,N.SUBJECT_ID,N.HADM_ID,N.CATEGORY,N.DESCRIPTION,N.CHARTDATE,N.TEXT
    FROM NOTEEVENTS N, (SELECT DISTINCT SUBJECT_ID,HADM_ID FROM PATIENT_VISIT) P
    WHERE (N.SUBJECT_ID,N.HADM_ID) = (P.SUBJECT_ID,P.HADM_ID)
    -- ISERROR - '1' INDICATES AN ERROR NOTE
    AND ISERROR IS NULL;

-- # of admissions
-- 17020 records
SELECT COUNT(1)
  FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
          FROM PATIENT_VISIT_NOTE)T;

-- distribution of category
-- 15 records
/*   category      |   cnt
-------------------+---------
Nursing/other     | 136489
Radiology         | 110864
Nursing           |  78765
Physician         |  51750
ECG               |  49004
Discharge summary |  18937
Echo              |  10212
Respiratory       |  10101
Nutrition         |   3205
General           |   2636
Rehab Services    |   1713
Social Work       |    700
Case Management   |    450
Consult           |     34
Pharmacy          |     30*/
SELECT CATEGORY,COUNT(1) AS CNT
  FROM PATIENT_VISIT_NOTE
  GROUP BY CATEGORY
  ORDER BY COUNT(1) DESC;

-- distribution of description
-- many records
/*                           description                              |   cnt
----------------------------------------------------------------------+---------
Report                                                               | 213032
Nursing Progress Note                                                |  66646
CHEST (PORTABLE AP)                                                  |  47991
Physician Resident Progress Note                                     |  24545
Nursing Transfer Note                                                |  11750*/
SELECT DESCRIPTION,COUNT(1) AS CNT
  FROM PATIENT_VISIT_NOTE
  GROUP BY DESCRIPTION
  ORDER BY COUNT(1) DESC;

-- distribution of 'Physician' category
-- many records
/*                   description                     |  cnt
----------------------------------------------------+--------
Physician Resident Progress Note                   | 24534
Physician Attending Progress Note                  |  8161
Intensivist Note                                   |  7441
Physician Resident Admission Note                  |  4273
Physician Attending Admission Note - MICU          |  1441
ICU Note - CVI                                     |  1199*/
SELECT DESCRIPTION,COUNT(1) AS CNT
  FROM PATIENT_VISIT_NOTE
  WHERE TRIM(CATEGORY) = 'Physician'
  GROUP BY DESCRIPTION
  ORDER BY COUNT(1) DESC;

SELECT *
  FROM PATIENT_VISIT_NOTE
  WHERE TRIM(CATEGORY) = 'Physician'
  AND TRIM(DESCRIPTION) = 'Physician Resident Admission Note'
  LIMIT 1;

-- distribution of 'Discharge summary' category
-- 2 records
/*  description |  cnt
-------------+--------
 Report      | 17327
 Addendum    |  1610*/
SELECT DESCRIPTION,COUNT(1) AS CNT
  FROM PATIENT_VISIT_NOTE
  WHERE TRIM(CATEGORY) = 'Discharge summary'
  GROUP BY DESCRIPTION
  ORDER BY COUNT(1) DESC;

SELECT *
  FROM PATIENT_VISIT_NOTE
  WHERE TRIM(CATEGORY) = 'Discharge summary'
  AND TRIM(DESCRIPTION) = 'Report'
  LIMIT 1;

-- # of (CATEGORY,DESCRIPTION) for each admission
/* cnt | count
-----+-------
   1 | 15863
   2 |   567
   3 |    92
   4 |    11
   5 |     2*/
SELECT CNT,COUNT(1) AS COUNT
FROM
  (SELECT SUBJECT_ID,HADM_ID,COUNT(1) AS CNT
    FROM PATIENT_VISIT_NOTE
    WHERE TRIM(CATEGORY) = 'Discharge summary'
    AND TRIM(DESCRIPTION) = 'Report'
    GROUP BY SUBJECT_ID,HADM_ID)T
  GROUP BY CNT
  ORDER BY COUNT(1) DESC;

-- # of (CATEGORY,DESCRIPTION) for each admission
/* cnt | count
-----+-------
   1 |   984
   2 |   697
   3 |   277
   4 |   114
   5 |    54
   6 |    17
   8 |    10
   7 |    10
   9 |     3
  10 |     2
  11 |     1
  16 |     1
  12 |     1*/
SELECT CNT,COUNT(1) AS COUNT
FROM
  (SELECT SUBJECT_ID,HADM_ID,COUNT(1) AS CNT
    FROM PATIENT_VISIT_NOTE
    WHERE TRIM(CATEGORY) = 'Physician'
    AND TRIM(DESCRIPTION) = 'Physician Resident Admission Note'
    GROUP BY SUBJECT_ID,HADM_ID)T
  GROUP BY CNT
  ORDER BY COUNT(1) DESC;

SELECT COUNT(1)
  FROM PATIENT_VISIT_NOTE T
  WHERE TRIM(CATEGORY) = 'Discharge summary'
  AND TRIM(DESCRIPTION) = 'Report';

-- # of admissions with aggregated notes
-- 16535 records
SELECT COUNT(1)
FROM
  (SELECT SUBJECT_ID, HADM_ID,
          STRING_AGG(TEXT::TEXT,',') AS NOTES
    FROM PATIENT_VISIT_NOTE T
    WHERE TRIM(CATEGORY) = 'Discharge summary'
    AND TRIM(DESCRIPTION) = 'Report'
    GROUP BY SUBJECT_ID, HADM_ID
    ORDER BY SUBJECT_ID, HADM_ID)T;

-- output notes data
COPY
  (SELECT SUBJECT_ID, HADM_ID,
          STRING_AGG(TEXT::TEXT,',') AS NOTES
    FROM PATIENT_VISIT_NOTE T
    WHERE TRIM(CATEGORY) = 'Discharge summary'
    AND TRIM(DESCRIPTION) = 'Report'
    GROUP BY SUBJECT_ID, HADM_ID
    ORDER BY SUBJECT_ID, HADM_ID)
   TO '/Users/Wenty/Documents/cs6140 Machine Learning/project/Predict Clinical Events/code/NOTES_DISCHARGE_SUMMARY_REPORT.CSV' WITH CSV HEADER;

---------------------------------------------------
-- distinct ICD9 code
-- 829 records
SELECT COUNT(DISTINCT ICD9_CODE) FROM PATIENT_VISIT;

-- update index for all 3-digit ICD codes
-- 829 records
DROP TABLE DAI_D_ICD;
CREATE TABLE DAI_D_ICD(
 ROW_ID SERIAL PRIMARY KEY,
 ICD9_CODE VARCHAR(10),
 CNT INT
);

INSERT INTO DAI_D_ICD(ICD9_CODE,CNT)
 SELECT ICD9_CODE,COUNT(1) AS CNT
   FROM PATIENT_VISIT
   GROUP BY ICD9_CODE
   ORDER BY CNT DESC;

-- frequent codes
-- 182 records
SELECT COUNT(1)
  FROM DAI_D_ICD
  WHERE CNT>171; -- 10% of # of admissions

-- 250 records
SELECT COUNT(1)
  FROM DAI_D_ICD
  WHERE CNT>85; -- 5% of # of admissions

---------------------------------------------------
-- transform PATIENT_VISIT to admissions with transformed ICD9 codes
-- 204103 records
DROP TABLE PATIENT_VISIT_CODE;
CREATE TABLE PATIENT_VISIT_CODE AS
  SELECT P.SUBJECT_ID, P.HADM_ID, P.ADMITTIME,D.ROW_ID AS ICD9_T
    FROM PATIENT_VISIT P
    INNER JOIN DAI_D_ICD D
    ON P.ICD9_CODE = D.ICD9_CODE
    ORDER BY P.SUBJECT_ID, P.HADM_ID, P.ADMITTIME,ICD9_T;

-- aggregate ICD9 code for each admission
-- 17130 records
DROP TABLE VISITS_FULL;
CREATE TABLE VISITS_FULL AS
  SELECT SUBJECT_ID, HADM_ID, ADMITTIME,
          STRING_AGG(ICD9_T::TEXT,',') AS ICD9_CODE
    FROM PATIENT_VISIT_CODE T
    GROUP BY SUBJECT_ID, HADM_ID, ADMITTIME
    ORDER BY SUBJECT_ID, ADMITTIME;

-- output data
COPY VISITS_FULL TO '/Users/Wenty/Documents/cs6140 Machine Learning/project/Predict Clinical Events/code/PATIENT_VISIT_FULL.CSV' WITH CSV HEADER;
---------------------------------------------------
-- transform PATIENT_VISIT to admissions with frequent ICD9 codes
-- 185696 records
DROP TABLE PATIENT_VISIT_FREQUENT_CODE;
CREATE TABLE PATIENT_VISIT_FREQUENT_CODE AS
  SELECT P.SUBJECT_ID, P.HADM_ID, P.ADMITTIME,D.ROW_ID AS ICD9_T
    FROM PATIENT_VISIT P
    INNER JOIN DAI_D_ICD D
    ON P.ICD9_CODE = D.ICD9_CODE
    -- set frequent code as appearing more than 10% of # of admissions
    -- WHERE D.CNT > 171
    WHERE D.CNT > 85 -- statistics below are for 171
    ORDER BY P.SUBJECT_ID, P.HADM_ID, P.ADMITTIME,ICD9_T;

-- delete patients with only 1 admission
-- 142 records
DELETE FROM PATIENT_VISIT_FREQUENT_CODE
WHERE SUBJECT_ID IN (SELECT SUBJECT_ID
                      FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
                              FROM PATIENT_VISIT_FREQUENT_CODE)T
                      GROUP BY SUBJECT_ID
                      HAVING COUNT(1) = 1);

-- statistics:
-- patients with more than 1 admissions
-- 6346 records
SELECT COUNT(DISTINCT SUBJECT_ID)
  FROM PATIENT_VISIT_FREQUENT_CODE;
-- # of admissions
-- 17053 records
SELECT COUNT(DISTINCT HADM_ID)
  FROM PATIENT_VISIT_FREQUENT_CODE;

-- average # of admissions
-- 2.69
SELECT AVG(CNT)
  FROM
  (SELECT SUBJECT_ID,COUNT(1) AS CNT
    FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
            FROM PATIENT_VISIT_FREQUENT_CODE)T
    GROUP BY SUBJECT_ID)C;

-- patients with # of admissions
-- 27 records
-- more than 5% of patients have [2,3,4] admissions
-- more than 1% of patients have [2,3,4,5,6] admissions
SELECT CNT,COUNT(1)
  FROM
  (SELECT SUBJECT_ID,COUNT(1) AS CNT
    FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
            FROM PATIENT_VISIT_FREQUENT_CODE)T
    GROUP BY SUBJECT_ID)C
  GROUP BY CNT
  ORDER BY COUNT(1) DESC;

-- average # of codes per admission
-- 10.88
SELECT AVG(CNT)
  FROM
  (SELECT SUBJECT_ID,HADM_ID,COUNT(1) AS CNT
    FROM PATIENT_VISIT_FREQUENT_CODE
    GROUP BY SUBJECT_ID,HADM_ID)C;

-- distribution of # of codes per admission
-- centered at [5,15]
SELECT CNT,COUNT(1)
  FROM
  (SELECT SUBJECT_ID,HADM_ID,COUNT(1) AS CNT
    FROM PATIENT_VISIT_FREQUENT_CODE
    GROUP BY SUBJECT_ID,HADM_ID)C
  GROUP BY CNT
  ORDER BY COUNT(1) DESC;

-- aggregate ICD9 code for each admission
-- 17085 records
DROP TABLE VISITS;
CREATE TABLE VISITS AS
  SELECT SUBJECT_ID, HADM_ID, ADMITTIME,
          STRING_AGG(ICD9_T::TEXT,',') AS ICD9_CODE
    FROM PATIENT_VISIT_FREQUENT_CODE T
    GROUP BY SUBJECT_ID, HADM_ID, ADMITTIME
    ORDER BY SUBJECT_ID, ADMITTIME;
SELECT * FROM VISITS LIMIT 10;
SELECT * FROM PATIENT_VISIT_FREQUENT_CODE LIMIT 30;

---------------------------------------------------
-- output data
COPY VISITS TO '/Users/Wenty/Documents/cs6140 Machine Learning/project/Predict Clinical Events/code/PATIENT_VISIT_85.CSV' WITH CSV HEADER;

COPY (SELECT CNT AS ADMISSION_LENGTH,COUNT(1) AS CNT
        FROM
        (SELECT SUBJECT_ID,COUNT(1) AS CNT
          FROM (SELECT DISTINCT SUBJECT_ID,HADM_ID
                  FROM PATIENT_VISIT_FREQUENT_CODE)T
          GROUP BY SUBJECT_ID)C
        GROUP BY CNT
        ORDER BY COUNT(1) DESC)
TO '/Users/Wenty/Documents/cs6140 Machine Learning/project/Predict Clinical Events/code/ADMISSION_DIST.CSV' WITH CSV HEADER;

COPY (SELECT CNT AS CODE_LENGTH,COUNT(1) AS CNT
        FROM
        (SELECT SUBJECT_ID,HADM_ID,COUNT(1) AS CNT
          FROM PATIENT_VISIT_FREQUENT_CODE
          GROUP BY SUBJECT_ID,HADM_ID)C
        GROUP BY CNT
        ORDER BY COUNT(1) DESC)
TO '/Users/Wenty/Documents/cs6140 Machine Learning/project/Predict Clinical Events/code/CODE_DIST.CSV' WITH CSV HEADER;
