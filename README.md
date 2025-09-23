# F1 Constructor Performance by month from 1958 to 2025

## What this project does
This project turns raw Formula 1 race results into easy-to-read monthly summaries for each team (constructor). It:
- reads the raw CSV files,
- cleans them and adds a month label,
- adds up each team’s points per month,
- compares a month to the previous month (Month-over-Month growth),
- makes a chart that shows the Top 10 teams for any month you pick (e.g. Aug 2012 or Apr 1993).

You can quickly see which team was strongest in a given month, and how their performance changed compared to the previous month.

### Examples of charts
You can find every month (since January 1958) chart preuploaded in reports/charts. 
Or you can use the [interactive webapp](https://github.com/nolimitbaiza/f1_constructor_performance_tracker/blob/13b233f31aaef891108e371537e3fb9cbac56885/reports/index.html) where you can choose what month performance you want to see. 

<img width="1280" height="800" alt="top10_1958-01" src="https://github.com/user-attachments/assets/0974e93f-c6e2-4539-8e79-c542b2ff9fe2" />
<img width="1280" height="800" alt="top10_2025-08" src="https://github.com/user-attachments/assets/e4fa0a99-f5c5-4269-bbb2-101447e6d834" />

### How to run
You can get all the charts by running 
```
python -m scripts.render_all_month
```

If you want a single month performance chart
```
python -m src.tracker.report --month 2012-08 --top 10
```
Change the '2012-08' and 'top 10' parts for the desired date and number of teams respectively.

### License & credits
Dataset: [F1 archive](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data/data) provided in data/raw/ 

License: MIT
