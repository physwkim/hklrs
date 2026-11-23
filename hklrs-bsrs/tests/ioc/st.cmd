# simDetector with one NDFileHDF5 plugin, for hklrs-bsrs/tests/rsm_ioc.rs.
# simdetector-ioc.sh sets TOP, ADCORE, ADSIMDETECTOR and PREFIX.
dbLoadDatabase("$(TOP)/dbd/simDetectorApp.dbd")
simDetectorApp_registerRecordDeviceDriver(pdbbase)
epicsEnvSet("EPICS_DB_INCLUDE_PATH", "$(ADCORE)/db")

# Frames up to 64 x 48; the test sets the size and data type it uses.
simDetectorConfig("SIM1", 64, 48, 1, 0, 0)
dbLoadRecords("$(ADSIMDETECTOR)/db/simDetector.template", "P=$(PREFIX),R=cam1:,PORT=SIM1,ADDR=0,TIMEOUT=1")

NDFileHDF5Configure("HDF1", 20, 0, "SIM1", 0)
dbLoadRecords("NDFileHDF5.template", "P=$(PREFIX),R=HDF1:,PORT=HDF1,ADDR=0,TIMEOUT=1,XMLSIZE=2048,NDARRAY_PORT=SIM1")

iocInit()
