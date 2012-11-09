import datetime

def print_results(attempts,hits):
    now = datetime.datetime.now()
    print '%s - Attempts: %d, hits: %d, error rate: %.5f%%' \
        %(now.strftime('%H:%M:%S'), attempts,hits,(1 - hits/float(attempts))*100)
