import numpy as np
# units: length [cm], time [s]

qpx = 0.09
qpy = 0.09
q_theta = 0.1

rpx = 0.1 
rpy = 0.1
rptheta = 0.01

Q = np.array([[qpx, 0, 0],[0, qpy, 0], [0, 0, q_theta]])
R = np.array([[rpx, 0, 0],[0, rpy, 0], [0, 0, rptheta]])


L = 9.95
k=np.pi/200
r = 2.2

def kalman_filter(wr, wl, camera, x_est_prev, P_est_prev, delta_T,
                  HT=None, HNT=None, RT=None, RNT=None):

    Ts = delta_T
    vr = k*r*wr
    vl = k*r*wl

    # State prediciton 
    x_est_a_priori = np.array([[x_est_prev[0][0] + Ts*np.cos(x_est_prev[2][0])*(vr+vl)/2], [x_est_prev[1][0] + Ts*np.sin(x_est_prev[2][0])*(vr+vl)/2], [x_est_prev[2][0] + Ts*(vr-vl)/L]])
    # Jacobian and covariance prediction
    Fk = np.array([[1, 0, -Ts*np.sin(x_est_prev[2][0])*(vr+vl)/2], [0, 1, Ts*np.cos(x_est_prev[2][0])*(vr+vl)/2], [0, 0, 1]])
    P_est_a_priori = np.dot(Fk, np.dot(P_est_prev, Fk.T)) + Q;
    

    # Update if we have the camera or return estimation
    if (camera is not None) :
        H = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
        i = camera - np.dot(H, x_est_a_priori);
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R;
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)));
        x_est = x_est_a_priori + np.dot(K,i);
        P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori));
        return x_est, P_est
    else:
        return x_est_a_priori, P_est_a_priori
