# Marketing Analytics Project - SQL & Python & Power BI & PowerPoint
The raw data for this project is presented in 6 CSV files. This data is from various sources, including customer reviews, social media comments, and campaign performance metrics, it represents the information of customers and products of ShopEasy. 

## Objective
ShopEasy, an online retail business, is facing reduced customer engagement and conversion rates despite launching several new online marketing campaigns. They need to conduct a detailed analysis and identify areas for improvement in their marketing strategies.

## Tools
|Tool        |Purpose                                                                        |
|------------|-------------------------------------------------------------------------------|
|SQL Server  |Cleaning the data                                       |
|Python|Doing the sentiment analysis|
|Mokkup AI   |Designing the wireframe/mockup of the dashboard                                |
|Power BI    |Visualizing the data via interactive dashboards                                |
|PowerPoint       |Calculate, explore the data, generate the findings and write the recommendation|
|GitHub      |Hosting the project documentation and version control                          |

## Data Cleaning with SQL
```
SELECT 
    c.CustomerID, 
    c.CustomerName,  
    c.Email,  
    c.Gender,
    c.Age,  
    g.Country, 
    g.City  
    dbo.customers as c  
LEFT JOIN
    dbo.geography g  
ON 
    c.GeographyID = g.GeographyID;  -

SELECT 
    ProductID, 
    ProductName,  
    Price,  
    CASE 
        WHEN Price < 50 THEN 'Low' 
        WHEN Price BETWEEN 50 AND 200 THEN 'Medium' 
        ELSE 'High' 
    END AS PriceCategory 
FROM 
    dbo.products

SELECT 
    ReviewID,  
    CustomerID, 
    ProductID,  
    ReviewDate,  
    Rating,  
    REPLACE(ReviewText, '  ', ' ') AS ReviewText
FROM 
    dbo.customer_reviews

SELECT 
    EngagementID,  
    ContentID,  
	CampaignID,  
    ProductID,  
    UPPER(REPLACE(ContentType, 'Socialmedia', 'Social Media')) AS ContentType,  
    LEFT(ViewsClicksCombined, CHARINDEX('-', ViewsClicksCombined) - 1) AS Views,  
	-- Extracts the Views part from the ViewsClicksCombined column by taking the substring before the '-' character
    RIGHT(ViewsClicksCombined, LEN(ViewsClicksCombined) - CHARINDEX('-', ViewsClicksCombined)) AS Clicks, 
	-- Extracts the Clicks part from the ViewsClicksCombined column by taking the substring after the '-' character
    Likes, 
    FORMAT(CONVERT(DATE, EngagementDate), 'dd.MM.yyyy') AS EngagementDate  -- Converts and formats the date as dd.mm.yyyy
FROM 
    dbo.engagement_data 
WHERE 
    ContentType != 'Newsletter' 

WITH DuplicateRecords AS (
    SELECT 
        JourneyID, 
        CustomerID, 
        ProductID,  
        VisitDate, 
        Stage,  
        Action, 
        Duration,  
        -- Use ROW_NUMBER() to assign a unique row number to each record within the partition defined below
        ROW_NUMBER() OVER (
            -- PARTITION BY groups the rows based on the specified columns that should be unique
            PARTITION BY CustomerID, ProductID, VisitDate, Stage, Action  
            -- ORDER BY defines how to order the rows within each partition (usually by a unique identifier like JourneyID)
            ORDER BY JourneyID  
        ) AS row_num  -- This creates a new column 'row_num' that numbers each row within its partition
    FROM 
        dbo.customer_journey  

-- Select all records from the CTE where row_num > 1, which indicates duplicate entries
    
SELECT *
FROM DuplicateRecords
WHERE row_num > 1  -- Filters out the first occurrence (row_num = 1) and only shows the duplicates (row_num > 1)
ORDER BY JourneyID

-- Outer query selects the final cleaned and standardized data
    
SELECT 
    JourneyID,  
    CustomerID,  
    ProductID,  
    VisitDate,  
    Stage,  
    Action, 
    COALESCE(Duration, avg_duration) AS Duration  
	-- Replaces missing durations with the average duration for the corresponding date
FROM 
    (
        -- Subquery to process and clean the data
        SELECT 
            JourneyID,  
            CustomerID,  
            ProductID,  
            VisitDate,  
            UPPER(Stage) AS Stage,  
            Action,  
            Duration, 
            AVG(Duration) OVER (PARTITION BY VisitDate) AS avg_duration,  
			-- Calculates the average duration for each date, using only numeric values
            ROW_NUMBER() OVER (
                PARTITION BY CustomerID, ProductID, VisitDate, UPPER(Stage), Action  
				-- Groups by these columns to identify duplicate records
                ORDER BY JourneyID  
				-- Orders by JourneyID to keep the first occurrence of each duplicate
            ) AS row_num 
			-- Assigns a row number to each row within the partition to identify duplicates
        FROM 
            dbo.customer_journey  
    ) AS subquery  -- Names the subquery for reference in the outer query
WHERE 
    row_num = 1;  
	-- Keeps only the first occurrence of each duplicate group identified in the subquery
```

## Sentiment Analysis with Python
```
# pip install pandas nltk pyodbc sqlalchemy
import pandas as pd
import pyodbc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon for sentiment analysis if not already present.
nltk.download('vader_lexicon')

# Define a function to fetch data from a SQL database using a SQL query
def fetch_data_from_sql():
    # Define the connection string with parameters for the database connection
    conn_str = (
        "Driver={SQL Server};"  # Specify the driver for SQL Server
        "Server=DESKTOP-LQNE95P;"  # Specify your SQL Server instance
        "Database=PortfolioProject_MarketingAnalytics;"  # Specify the database name
        "Trusted_Connection=yes;"  # Use Windows Authentication for the connection
    )
    # Establish the connection to the database
    conn = pyodbc.connect(conn_str)
    
    # Define the SQL query to fetch customer reviews data
    query = "SELECT ReviewID, CustomerID, ProductID, ReviewDate, Rating, ReviewText FROM customer_reviews"
    
    # Execute the query and fetch the data into a DataFrame
    df = pd.read_sql(query, conn)
    
    # Close the connection to free up resources
    conn.close()
    
    # Return the fetched data as a DataFrame
    return df

# Fetch the customer reviews data from the SQL database
customer_reviews_df = fetch_data_from_sql()

# Initialize the VADER sentiment intensity analyzer for analyzing the sentiment of text data
sia = SentimentIntensityAnalyzer()

# Define a function to calculate sentiment scores using VADER
def calculate_sentiment(review):
    # Get the sentiment scores for the review text
    sentiment = sia.polarity_scores(review)
    # Return the compound score, which is a normalized score between -1 (most negative) and 1 (most positive)
    return sentiment['compound']

# Define a function to categorize sentiment using both the sentiment score and the review rating
def categorize_sentiment(score, rating):
    # Use both the text sentiment score and the numerical rating to determine sentiment category
    if score > 0.05:  # Positive sentiment score
        if rating >= 4:
            return 'Positive'  # High rating and positive sentiment
        elif rating == 3:
            return 'Mixed Positive'  # Neutral rating but positive sentiment
        else:
            return 'Mixed Negative'  # Low rating but positive sentiment
    elif score < -0.05:  # Negative sentiment score
        if rating <= 2:
            return 'Negative'  # Low rating and negative sentiment
        elif rating == 3:
            return 'Mixed Negative'  # Neutral rating but negative sentiment
        else:
            return 'Mixed Positive'  # High rating but negative sentiment
    else:  # Neutral sentiment score
        if rating >= 4:
            return 'Positive'  # High rating with neutral sentiment
        elif rating <= 2:
            return 'Negative'  # Low rating with neutral sentiment
        else:
            return 'Neutral'  # Neutral rating and neutral sentiment

# Define a function to bucket sentiment scores into text ranges
def sentiment_bucket(score):
    if score >= 0.5:
        return '0.5 to 1.0'  # Strongly positive sentiment
    elif 0.0 <= score < 0.5:
        return '0.0 to 0.49'  # Mildly positive sentiment
    elif -0.5 <= score < 0.0:
        return '-0.49 to 0.0'  # Mildly negative sentiment
    else:
        return '-1.0 to -0.5'  # Strongly negative sentiment

# Apply sentiment analysis to calculate sentiment scores for each review
customer_reviews_df['SentimentScore'] = customer_reviews_df['ReviewText'].apply(calculate_sentiment)

# Apply sentiment categorization using both text and rating
customer_reviews_df['SentimentCategory'] = customer_reviews_df.apply(
    lambda row: categorize_sentiment(row['SentimentScore'], row['Rating']), axis=1)

# Apply sentiment bucketing to categorize scores into defined ranges
customer_reviews_df['SentimentBucket'] = customer_reviews_df['SentimentScore'].apply(sentiment_bucket)

# Display the first few rows of the DataFrame with sentiment scores, categories, and buckets
print(customer_reviews_df.head())

# Save the DataFrame with sentiment scores, categories, and buckets to a new CSV file
customer_reviews_df.to_csv('fact_customer_reviews_with_sentiment.csv', index=False)

# Find the new file location
# import os
# current_directory = os.getcwd()
# print(current_directory)
```

## Visualization
![Image](https://github.com/user-attachments/assets/f0497ce5-4d2a-4b47-86e5-3227cdc3980f)

## Actions
#### Increase Conversion Rates:
Target High-Performing Product Categories: Focus marketing efforts on products with demonstrated high conversion rates, such as Kayaks, Ski Boots, and Baseball Gloves. Implement seasonal promotions or personalized campaigns during peak months (e.g., January and September) to capitalize on these trends.
#### Enhance Customer Engagement:
Revitalize Content Strategy: To turn around declining views and low interaction rates, experiment with more engaging content formats, such as interactive videos or user-generated content. Additionally, boost engagement by optimizing call-to-action placement in social media and blog content, particularly during historically lower-engagement months (September-December).

## Presentation
Presenting this project on PowerPoint.
![Image](https://github.com/user-attachments/assets/fa4898f1-3198-44fa-93d4-6366bceb2697)
