import os
import subprocess
from datetime import datetime

def timenow():
    return str(datetime.now().isoformat())
def squeue_check(ID, user = 'guillermo.vazquez', out_print = False):
    cmd = ['squeue', '-h', '-u', user, '-j', str(ID)]
    squeue_output=subprocess.run(cmd,capture_output=True)
    
    if squeue_output.stdout==b'':
        print('ERROR in squeue check: '+squeue_output.stderr.decode("utf-8") )
        status = 'DONE'
    else:
        out_work=squeue_output.stdout.split()
        if len(out_work)>6:
            
            job_name=  out_work[1].decode("utf-8")
            
            partition= out_work[3].decode("utf-8")
            nodes=     out_work[4].decode("utf-8")
            tasks=     out_work[5].decode("utf-8")
            status=    out_work[6].decode("utf-8")
            time =     out_work[7].decode("utf-8")
            time_left= out_work[8].decode("utf-8")
            start_time=out_work[9].decode("utf-8")
    if out_print:
        if status=='RUNNING':
            print('at '+timenow()+' :'+' this job is been running for '+time+ ' since '+start_time+' and there is '+
                        time_left+ ' time left' )
        if status=='PENDING':
            print('at '+timenow()+' :'+' this job is predicted to start at '+start_time)
    return status



def squeue_check_av(ID, user = 'guillermo.vazquez', out_print = False):
    cmd = ['sacct', '-j', str(ID)]
    squeue_output=subprocess.run(cmd, capture_output=True)
    out_work=squeue_output.stdout.split()

    job_id =   out_work[22].decode("utf-8")
    nodes=     out_work[26].decode("utf-8")
    tasks=     out_work[25].decode("utf-8")
    time =     out_work[28].decode("utf-8")
    start_time=out_work[29].decode("utf-8")
    status =   out_work[27].decode("utf-8")
    
    # print('job_id', job_id)
    # print('nodes', nodes)
    # print('tasks', tasks)
    # print('time', time)
    # print('start_time', start_time)
    # print('status', status)
    return status, time

    # if squeue_output.stdout==b'':
    #     print('ERROR in squeue check: '+squeue_output.stderr.decode("utf-8") )
    #     status = 'DONE'
    # else:
    #     out_work=squeue_output.stdout.split()
    #     if len(out_work)>6:
            
    #         job_name=  out_work[1].decode("utf-8")
            
    #         partition= out_work[3].decode("utf-8")
    #         nodes=     out_work[4].decode("utf-8")
    #         tasks=     out_work[5].decode("utf-8")
    #         status=    out_work[6].decode("utf-8")
    #         time =     out_work[7].decode("utf-8")
    #         time_left= out_work[8].decode("utf-8")
    #         start_time=out_work[9].decode("utf-8")
    # if out_print:
    #     if status=='RUNNING':
    #         print('at '+timenow()+' :'+' this job is been running for '+time+ ' since '+start_time+' and there is '+
    #                     time_left+ ' time left' )
    #     if status=='PENDING':
    #         print('at '+timenow()+' :'+' this job is predicted to start at '+start_time)
    # return status