std_msgs/Header header
int32 id
int32 counter

float64 meas_x
float64 meas_y
float64 meas_v
float64 meas_phi

float64 pred_x
float64 pred_y
float64 pred_v
float64 pred_phi

float64 post_x
float64 post_y
float64 post_v
float64 post_phi

float64 innov_x
float64 innov_y
float64 innov_v
float64 innov_phi

float64[4] p_pred_diag
float64[4] p_post_diag
float64[4] q_diag
float64[4] r_diag

float64 nis