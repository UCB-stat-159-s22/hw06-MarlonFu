# Generate summary table.

.PHONY : clean
clean :
	rm -f audio/*
    rm -f figures/*
    rm -f _build/*
   
    
.PHONY : all
all:
	jupyter execute index.ipynb

