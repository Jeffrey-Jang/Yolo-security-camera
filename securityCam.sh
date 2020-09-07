#!/bin/bash
CAMHOMEPATH=/home/pi/Yolo-security-camera
TDATE=`date '+%y%m%d_%H%M%S'`
LOGPATH=${CAMHOMEPATH}/log
LOGFILE=${LOGPATH}/log_SecurityCam.log
CAMPROC=${CAMHOMEPATH}/record
OUTPATH=${CAMPROC}/${TDATE}


if [ $1 = "true" ]
then
	cd ${CAMHOMEPATH}
    mkdir ${OUTPATH}
    mkdir ${OUTPATH}/detacted
    echo "[`date '+%y/%m/%d %H:%M:%S'`] ON" >> ${LOGFILE}
    python3 ${CAMHOMEPATH}/yolo_video.py --input /dev/video0 --output ${OUTPATH}/ >> ${LOGFILE} 2>&1

elif [ $1 = "false" ]
then
	if test -e "${CAMPROC}/cam_process.txt"; then #
		echo "Kill the SecurityCam..." >> ${LOGFILE}
		touch ${CAMPROC}/CAM_STOP
		#kill -9 `cat ${CAMPROC}/cam_process.txt`
		rm -f ${CAMPROC}/cam_process.txt
		echo "[`date '+%y/%m/%d %H:%M:%S'`] OFF" >> ${LOGFILE}
	else
		echo "cam_process.txt NOT found." >> ${LOGFILE}
		echo "Trying kill the SecurityCam, but it is idle." >> ${LOGFILE}
		echo "[`date '+%y/%m/%d %H:%M:%S'`] IDLE" >> ${LOGFILE}
	fi
fi
	
