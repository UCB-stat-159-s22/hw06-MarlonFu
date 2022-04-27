# Generate summary table.

.PHONY: env
env:
	mamba env create -f environment.yml

.PHONY: html
html:
	jupyterbook build .

.PHONY : clean
clean :
	rm -f audio/*.eav
    rm -f figures/*.png
    rm -rf _build
	rm -rf jupyterbook/_build
   
    
.PHONY : all
all:
	jupyter execute index.ipynb

