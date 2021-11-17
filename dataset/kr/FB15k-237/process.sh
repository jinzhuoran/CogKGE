# cat run.log | sed -E 's/\[.*\]\[.*\]\[.*\]\[.*\]\[(.*)\]/\1/'
cd ./experimental_output
for file in $(ls)
do  
    if [ -d "$file" ];then
        # echo $file 
        logfile="$file/run.log"
        # echo $logfile
        # cat $logfile
        cat $logfile | sed -E 's/\[.*\]\[.*\]\[.*\]\[.*\]\[(.*)\]/\1/' > "$file/simple_run.log"
        echo "Generate $file/simple_run.log!"
    fi
done
