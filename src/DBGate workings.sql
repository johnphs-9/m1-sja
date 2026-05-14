-- ALTER TABLE sg_job_data
ADD PRIMARY KEY (listing_id);

DROP TABLE job_listing_categories;
DROP TABLE categories;

CREATE OR REPLACE TABLE categories (
    category_id INTEGER NOT NULL PRIMARY KEY,
    category_name VARCHAR NOT NULL
);

-- populating the categories table
INSERT INTO categories (category_id, category_name)
SELECT DISTINCT
    CAST(json_extract_string(cat.value, '$.id') AS INTEGER) AS category_id,
    json_extract_string(cat.value, '$.category') AS category_name
FROM sg_job_data j,
     json_each(j.categories::JSON) AS cat
ORDER BY category_id;


-- creating the bridging TABLE
CREATE OR REPLACE TABLE job_listing_categories (
    listing_id BIGINT NOT NULL,
    category_id INTEGER NOT NULL,
    PRIMARY KEY (listing_id, category_id),
    FOREIGN KEY (listing_id)
        REFERENCES sg_job_data(listing_id),
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
);


-- ============================================================
-- 4. Insert job-category relationships
-- ============================================================
INSERT INTO job_listing_categories (listing_id, category_id) 
SELECT DISTINCT
    j.listing_id,
    CAST(json_extract_string(cat.value, '$.id') AS INTEGER) AS category_id
FROM sg_job_data j, 
    json_each(j.categories::JSON) AS cat;


-- ALTER TABLE sg_job_data DROP COLUMN categories;

-- there are nulls in metadata_jobPostId
SELECT * 
FROM sg_job_data jobs WHERE jobs.metadata_jobPostId IS NULL
ORDER by listing_id;

SELECT COUNT(DISTINCT listing_id) FROM job_listing_categories listcat WHERE listcat.listing_id IN 
(SELECT listing_id FROM sg_job_data jobs where jobs.metadata_jobPostId IS NOT NULL)
;
-- ORDER BY listcat.listing_id;


SELECT COUNT(DISTINCT listing_id) FROM job_listing_categories;
SELECT count(listing_id) FROM sg_job_data WHERE metadata_jobPostId IS NOT NULL;