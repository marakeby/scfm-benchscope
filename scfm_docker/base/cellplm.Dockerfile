FROM scfm-base:cu128
RUN pip install cellplm
RUN python -c "import CellPLM; print('cellplm ok')"