import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import threading


class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.bridge = CvBridge()

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.running = True

        self.spin_thread = threading.Thread(
            target=self.spin_thread_func
        )
        self.spin_thread.start()

    def spin_thread_func(self):

        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):

        with self.frame_lock:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                "bgr8"
            )

    def stop(self):

        self.running = False
        self.spin_thread.join()

    def process_image(self, img):

        msg = Twist()

        rows, cols = img.shape[:2]

        R = img[:, :, 2]

        redMask = np.zeros_like(R)
        redMask[(R >= 220) & (R <= 255)] = 255

        contours, _ = cv2.findContours(
            redMask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:

            c = max(contours, key=cv2.contourArea)

            M = cv2.moments(c)

            if M["m00"] != 0:

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(
                    img,
                    (cx, cy),
                    10,
                    (0, 255, 0),
                    -1
                )

                area = cv2.contourArea(c)

                print("Ball Area:", area)

                STOP_AREA = 15000

                if area > STOP_AREA:

                    msg.linear.x = 0.0
                    msg.angular.z = 0.0

                elif abs(cols / 2 - cx) > 20:

                    msg.linear.x = 0.0

                    if cols / 2 > cx:
                        msg.angular.z = 0.2
                    else:
                        msg.angular.z = -0.2

                else:

                    msg.linear.x = 0.2
                    msg.angular.z = 0.0

        else:

            msg.linear.x = 0.0
            msg.angular.z = 0.15

        self.publisher.publish(msg)

        return img

    def display_image(self):

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 800, 600)

        while rclpy.ok():

            if self.latest_frame is not None:

                result = self.process_image(
                    self.latest_frame.copy()
                )

                cv2.imshow(
                    "frame",
                    result
                )

                self.latest_frame = None

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()


def main(args=None):

    print("OpenCV version:", cv2.__version__)

    rclpy.init(args=args)

    node = ImageSubscriber()

    try:
        node.display_image()

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
