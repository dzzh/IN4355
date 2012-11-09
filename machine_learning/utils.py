def print_results(attempts,hits):
    print 'Attempts: %d, hits: %d, error rate: %.5f%%' %(attempts,hits,(1 - hits/float(attempts))*100)
