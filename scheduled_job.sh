#!/bin/sh

# Run the Reddit bot script every night at 11PM
# Set up to be used with a crontab
# 30 23 * * * /Users/alexandradelucia/storybot/sheduled_job.sh >> /Users/alexandradelucia/storybot/logs/crontab.log 2>&1

cd /Users/alexandradelucia/storybot

NOW=$(date +"%Y_%m_%d")
LOG_FILE="logs/$NOW.out"

# Run script
./bot.py --log "$LOG_FILE"

# Check for success
status=$?
if [ $status -ne 0 ]
then
    echo "Job failure on $NOW. Check $LOG_FILE for more information."
else
    echo "Job success on $NOW."
fi

