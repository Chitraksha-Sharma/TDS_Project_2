# README.md

## Dataset Analysis Overview

### Dataset Location
The dataset is located at:
```
C:\Users\Regal\Desktop\IITM\DS_Tools\project_2\media\media.csv
```

### Dataset Structure
The dataset contains the following columns with their respective data types:

- **date**: Object (O)
- **language**: Object (O)
- **type**: Object (O)
- **title**: Object (O)
- **by**: Object (O)
- **overall**: Integer (int64)
- **quality**: Integer (int64)
- **repeatability**: Integer (int64)
- **Outlier**: Integer (int64)
- **Cluster**: Float (float64)

### Summary Statistics
- Total Entries: 2,652
- Unique Dates: 2,055
- Unique Languages: 11
- Unique Types: 8
- Unique Titles: 2,312
- Unique Contributors: 1,528
- Missing Values: 
  - `date`: 99 entries
  - `by`: 262 entries

#### Statistical Overview
- **Overall Ratings**: Mean = 3.05, Range = 1 to 5
- **Quality Ratings**: Mean = 3.21, Range = 1 to 5
- **Repeatability Ratings**: Mean = 1.49, Range = 1 to 3

### Analysis Performed
The analysis focused on the following aspects:
- Missing value detection
- Correlation analysis
- Outlier detection
- Feature importance evaluation

### Key Insights and Implications

#### Data Composition
1. The dataset contains diverse entries across various columns, heavily featuring unique titles and contributors.
  
#### Missing Values
2. Missing entries in the `date` and `by` columns suggest potential data quality issues that need to be addressed.

#### Statistical Overview
3. The ratings data indicates a general trend of moderate ratings with `overall` and `quality` values averaging around 3, while `repeatability` is more limited.

#### Correlation Insights
4. Strong positive correlation (0.826) between `overall` and `quality` ratings signifies an important relationship between these metrics.
5. Moderate correlation (0.513) between `repeatability` and `overall` indicates that repeated viewings may positively impact ratings.

#### Feature Importance
6. `Quality` has the highest impact on overall ratings (35.9%), followed by `overall` itself (26.8%) and `repeatability` (21.6%).

### Actionable Implications
1. **Data Cleaning**: Implement strategies to address missing values in `date` and `by` columns to enhance data quality.
2. **Quality Focus**: Media producers should prioritize quality improvements based on insights from high-quality rated content.
3. **Engagement Strategies**: Encourage repeat viewings through various engagement strategies to boost overall ratings.
4. **Content Selection**: Use data on frequently rated titles to inform content acquisition and production strategies.
5. **Targeted Marketing**: Leverage demographic insights on successful titles and contributors for more effective marketing campaigns.
6. **Outlier Monitoring**: Investigate the `Outlier` column to refine evaluation criteria and understand anomalies in ratings.
7. **Clustering Utilization**: Further explore the `Cluster` feature to identify segments of high-performing content, guiding strategic decisions.

By leveraging the insights derived from this dataset, stakeholders can enhance media quality, viewer engagement, and overall performance in the entertainment landscape.