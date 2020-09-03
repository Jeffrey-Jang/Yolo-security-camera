#!/bin/bash
TDATE=`date '+%y%m%d_%H%M%S'`
LOGPATH=/home/pi/skills/log
LOGFILE=${LOGPATH}/log_SecurityCam.log
CAMPROC=/home/pi/skills/record
OUTPATH=${CAMPROC}/${TDATE}


if [ $1 = "true" ]
then
	cd /home/pi/keras-yolo3
    mkdir ${OUTPATH}
    mkdir ${OUTPATH}/detacted
    echo "[`date '+%y/%m/%d %H:%M:%S'`] ON" >> ${LOGFILE}
    python3 /home/pi/keras-yolo3/yolo_video.py --input /dev/video0 --output ${OUTPATH}/ >> ${LOGFILE} 2>&1

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
	