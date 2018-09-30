LENS_SRC= ${LENSDIR}/Src
LENS_BIN= ${LENSDIR}/Bin/${HOSTTYPE}

# comment/uncomment depending on which version of Tcl/Tk you use
TCLVER= 8.3.4
LIBS= -L${LENS_BIN} -llens2.63 -ltk8.3 -ltcl8.3 -lm -lX11
#TCLVER= 8.5.10
#LIBS= -L${LENS_BIN} -llens2.63 -ltk8.5 -ltcl8.5 -lm -lX11

TCLDIR= ${LENSDIR}/TclTk
INCL=   -I$(TCLDIR)/tcl$(TCLVER)/generic -I$(TCLDIR)/tk$(TCLVER)/generic \
	-I$(TCLDIR)/tcl$(TCLVER)/unix -I$(TCLDIR)/tk$(TCLVER)/unix \
	-I/home/rmintz/igraph-0.7.1/include

social: social.c Makefile
	gcc -Wall -o social -I${LENS_SRC} social.c ${INCL} ${LIBS} -L/home/rmintz/igraph-0.7.1/src/.libs -ligraph
