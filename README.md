# 8-Puzzle AI Solver với Pygame

Dự án này là một trình giải câu đố 8-puzzle (8-Puzzle) trực quan được triển khai bằng Python sử dụng thư viện Pygame. Nó cho phép người dùng thiết lập một trạng thái 8-Puzzle, chọn từ nhiều thuật toán tìm kiếm AI khác nhau và trực quan hóa quá trình giải đố. Dự án cũng bao gồm chế độ benchmark để so sánh hiệu suất của các thuật toán tìm đường khác nhau.

## Mục lục

- [Tính năng](#tính-năng)
- [Các Thuật Toán Được Triển Khai](#các-thuật-toán-được-triển-khai)
- [Yêu Cầu Cài Đặt](#yêu-cầu-cài-đặt)
- [Cách Chạy](#cách-chạy)
- [Cách Sử Dụng](#cách-sử-dụng)
  - [Nhập Câu Đố](#nhập-câu-đố)
  - [Giải Câu Đố](#giải-câu-đố)
  - [Xem Kết Quả](#xem-kết-quả)
  - [Điều Khiển](#điều-khiển)
  - [Ghi Chú](#ghi-chú)
- [GIF ghi lại cách hoạt động của các thuật toán](#gif-ghi-lại-cách-hoạt-động-của-các-thuật-toán)
- [Tổng Quan Cấu Trúc Code](#tổng-quan-cấu-trúc-code)
- [Những Cải Tiến Tiềm Năng](#những-cải-tiến-tiềm-năng)

## Tính năng

*   **Giao diện người dùng đồ họa (GUI)** tương tác dựa trên Pygame.
*   Nhập câu đố thủ công bằng cách nhấp vào các ô trên lưới.
*   Tự động tạo các câu đố ngẫu nhiên có thể giải được (sử dụng bởi một số thuật toán như "Backtrack","CSP Slover).
*   **Trực quan hóa** quá trình tìm kiếm hoặc giải quyết cho hầu hết các thuật toán.
*   Triển khai nhiều thuật toán tìm kiếm AI (xem danh sách bên dưới).
*   Hiển thị kết quả chi tiết: đường đi, thời gian thực thi, số bước/trạng thái đã khám phá, và các nước đi cụ thể.
*   **Chế độ Benchmark** để so sánh hiệu suất của các thuật toán tìm đường trên cùng một câu đố.
*   Có thể điều chỉnh tốc độ trực quan hóa và độ sâu tối đa cho các thuật toán liên quan (IDDFS, DLS Visit).
*   Thông báo lỗi và trạng thái rõ ràng cho người dùng.
*   Sử dụng threading để chạy các thuật toán tốn thời gian mà không làm đóng băng GUI.

## Các Thuật Toán Được Triển Khai

Ứng dụng triển khai một loạt các thuật toán tìm kiếm và giải quyết vấn đề:

1.  **Tìm kiếm Không Thông tin (Uninformed Search):**
    *   `BFS` (Breadth-First Search - Tìm kiếm theo chiều rộng)
    *   `UCS` (Uniform Cost Search - Tìm kiếm chi phí đồng nhất)
    *   `DFS` (Depth-First Search - Tìm kiếm theo chiều sâu)
    *   `IDDFS` (Iterative Deepening Depth-First Search - Tìm kiếm sâu dần lặp)

2.  **Tìm kiếm Thông tin (Informed/Heuristic Search):**
    *   `Greedy` (Greedy Best-First Search - Tìm kiếm tham lam tốt nhất đầu tiên)
    *   `A*` (A-Star Search)
    *   `IDA*` (Iterative Deepening A-Star Search)

3.  **Tìm kiếm Cục bộ & Siêu Heuristic (Local Search & Metaheuristics):**
    *   `Spl HC` (Simple Hill Climbing - Leo đồi đơn giản)
    *   `Stee HC` (Steepest Ascent Hill Climbing - Leo đồi dốc nhất)
    *   `Stoch HC` (Stochastic Hill Climbing - Leo đồi ngẫu nhiên)
    *   `SA` (Simulated Annealing - Luyện kim mô phỏng)
    *   `Genetic` (Genetic Algorithm - Thuật toán di truyền)
    *   `Beam` (Beam Search - Tìm kiếm chùm)
      
4. ** Các thuật toán CSPs:**
    *   `Backtrack` (Quay lui - tương tự DFS, tạo và giải hoặc chỉ khám phá)
    *   `CSP Solve` (Giải bài toán ràng buộc - triển khai dưới dạng tìm kiếm quay lui)

5.  **Các Thuật Toán Complex Enviroment:**

    *   `And-or-search` (tìm đường đi ngắn nhất trong giới hạn độ sâu)
    *   `Belief` (Belief State Search - Tìm kiếm không gian niềm tin)

6.  **Các Thuật Toán RL**
    *   `Q-Learning` (Thuật toán học tăng cường Q-Learning)

## Yêu Cầu Cài Đặt

1.  **Python 3.x** (khuyến nghị 3.7 trở lên).
2.  **Pygame**:
    pip install pygame
3.  **NumPy** (bắt buộc cho Q-Learning):
    pip install numpy

## Cách Chạy

1.  Đảm bảo bạn đã cài đặt Python, Pygame và NumPy.
2.  Lưu toàn bộ đoạn mã trên vào một file Python (ví dụ: `puzzle_solver.py`).
3.  Mở terminal hoặc command prompt, điều hướng đến thư mục chứa file.
4.  Chạy file bằng lệnh:
    python 23133061_PhanTrongQui_tuan13.py

## Cách Sử Dụng

Giao diện người dùng được chia thành ba phần chính: Lưới câu đố (trái), Bảng điều khiển (giữa), và Bảng kết quả (phải). Một bảng điều khiển phụ ở dưới cùng chứa các thanh trượt và ghi chú.

### Nhập Câu Đố

1.  **Thủ công**:
    *   Khi ứng dụng khởi động, bạn sẽ thấy một lưới 3x3 trống.
    *   Nhấp vào một ô trống để đặt số `1`.
    *   Tiếp tục nhấp vào các ô trống khác để đặt các số `2` đến `8`.
    *   Ô cuối cùng còn lại sẽ tự động là `0` (ô trống).
    *   Sau khi đặt đủ 8 số, ứng dụng sẽ kiểm tra xem câu đố có thể giải được không.
        *   Nếu có thể giải, các nút thuật toán sẽ được kích hoạt.
        *   Nếu không thể giải, một thông báo cảnh báo sẽ xuất hiện; bạn cần nhấn `Reset`.
2.  **Tự động (cho một số thuật toán)**:
    *   Các thuật toán như `Backtrack` hoặc `CSP Solve` sẽ tự động tạo một câu đố ngẫu nhiên có thể giải được và sau đó cố gắng giải nó.

### Giải Câu Đố

1.  Sau khi một câu đố hợp lệ và có thể giải được đã được thiết lập (hoặc nếu bạn chọn một thuật toán tự tạo câu đố), các nút thuật toán tương ứng trong Bảng điều khiển sẽ được kích hoạt.
2.  Nhấp vào nút của thuật toán bạn muốn sử dụng.
3.  Ứng dụng sẽ hiển thị thông báo "Running..." và lưới có thể tạm thời trống hoặc hiển thị trạng thái bắt đầu.
4.  Sau khi thuật toán hoàn tất, quá trình giải (hoặc khám phá) sẽ được trực quan hóa trên lưới.

### Xem Kết Quả

*   **Bảng Kết Quả** (bên phải) sẽ hiển thị thông tin chi tiết về quá trình giải, bao gồm:
    *   Thuật toán đã sử dụng và trạng thái (đã giải, thất bại, v.v.).
    *   Thời gian thực thi.
    *   Số bước trong đường đi giải pháp (nếu tìm thấy).
    *   Đối với các thuật toán tìm đường thành công, một danh sách các nước đi để giải câu đố sẽ được hiển thị.
    *   Đối với Q-Learning, các chỉ số thống kê bổ sung sẽ được hiển thị.
*   Nếu kết quả quá dài, bạn có thể sử dụng con lăn chuột khi trỏ vào bảng kết quả để cuộn.

### Điều Khiển

*   **Bảng Điều Khiển (Giữa):**
    *   Chứa các nút để chọn thuật toán.
    *   Nút `Benchmark`: Chạy một tập hợp các thuật toán tìm đường trên câu đố hiện tại và so sánh kết quả của chúng.
    *   Nút `Reset`: Xóa lưới hiện tại, kết quả, và đặt lại ứng dụng về trạng thái ban đầu để nhập câu đố mới.
*   **Bảng Điều Khiển Phụ (Dưới Cùng, bên trái Bảng Kết Quả):**
    *   **Vis Speed Slider**: Điều chỉnh tốc độ của hoạt ảnh trực quan hóa (tính bằng mili giây mỗi bước). Giá trị nhỏ hơn = nhanh hơn.
    *   **Max Depth Slider**: Đặt giới hạn độ sâu tối đa cho các thuật toán `IDDFS` và `DLS Visit` (And-Or Search).

### Ghi Chú

*   Khu vực phía dưới bên phải (dưới bảng kết quả, bên phải các thanh trượt) hiển thị một số ghi chú và mẹo nhanh.

## Tổng Quan Cấu Trúc Code

File mã nguồn chính (`puzzle_solver.py`) được cấu trúc như sau:

1.  **Import và Khởi tạo Pygame**:
    *   Nhập các thư viện cần thiết.
    *   Khởi tạo Pygame và các module của nó (đặc biệt là `pygame.font`).

2.  **Hằng số Toàn cục**:
    *   Kích thước màn hình, lưới, ô, panel (`GRID_SIZE`, `TILE_SIZE`, `WIDTH`, `HEIGHT`, v.v.).
    *   Màu sắc (`WHITE`, `BLACK`, `RED`, `TILE_COLORS`, v.v.).
    *   Font chữ (`TILE_FONT`, `BUTTON_FONT`, v.v.).
    *   Trạng thái đích (`GOAL_STATE`).
    *   Các giới hạn mặc định cho thuật toán (`DEFAULT_MAX_DEPTH`, `MAX_BFS_STATES`, v.v.).

3.  **Hàm Hỗ trợ (Helper Functions)**:
    *   `state_to_tuple()`: Chuyển đổi trạng thái list-of-lists thành tuple-of-tuples.
    *   `is_valid_state()`: Kiểm tra tính hợp lệ của một trạng thái câu đố.
    *   `get_inversions()`, `is_solvable()`: Xác định xem một trạng thái câu đố có thể giải được không.
    *   `generate_random_solvable_state()`: Tạo một trạng thái câu đố ngẫu nhiên và có thể giải được.
    *   `get_solution_moves()`: Chuyển đổi một đường đi các trạng thái thành một danh sách các nước đi dễ đọc.

4.  **Lớp `PuzzleState`**:
    *   Đại diện cho một trạng thái của câu đố.
    *   Chủ yếu được sử dụng bởi thuật toán Q-Learning và để đảm bảo tính nhất quán trong việc xử lý trạng thái.
    *   Bao gồm các phương thức như `__init__`, `__eq__`, `__hash__`, `get_children`.

5.  **Lớp `Puzzle`**:
    *   Chứa logic cốt lõi của bài toán 8-Puzzle:
        *   `__init__()`: Khởi tạo với trạng thái bắt đầu và đích.
        *   `get_blank_position()`: Tìm vị trí ô trống.
        *   `is_goal()`: Kiểm tra xem một trạng thái có phải là đích không.
        *   `get_neighbors()`: Lấy các trạng thái lân cận hợp lệ.
        *   `heuristic()`: Hàm heuristic Manhattan distance (mặc định).
        *   `get_misplaced_tiles_heuristic()`: Hàm heuristic số ô sai vị trí.
    *   **Triển khai tất cả các thuật toán tìm kiếm** dưới dạng các phương thức của lớp này (ví dụ: `bfs()`, `a_star()`, `simulated_annealing()`, `q_learning()`, v.v.). Mỗi phương thức trả về đường đi (list các trạng thái), một trạng thái đơn lẻ, hoặc một cấu trúc dữ liệu tùy chỉnh (ví dụ: dict cho Q-Learning).

6.  **Lớp `PygamePuzzleApp`**:
    *   Quản lý cửa sổ Pygame, vòng lặp sự kiện, và tất cả các yếu tố GUI.
    *   `__init__()`: Thiết lập màn hình, đồng hồ, trạng thái ứng dụng, và các thành phần UI.
    *   `_create_ui_elements()`: Tạo các nút và thanh trượt.
    *   **Phương thức Vẽ (`draw_...`)**: `draw_grid()`, `draw_buttons()`, `draw_message()`, `draw_results()`, `draw_sliders()`, `draw_notes()`.
    *   **Xử lý Trạng thái và Logic Ứng dụng**:
        *   `update_button_states()`: Kích hoạt/vô hiệu hóa các nút dựa trên trạng thái hiện tại.
        *   `handle_input()`: Xử lý các sự kiện chuột và bàn phím.
        *   `_handle_grid_click()`, `_handle_button_click()`: Logic cụ thể cho các tương tác.
        *   `reset_app()`: Đặt lại ứng dụng.
        *   `start_random_placement()`, `_update_placement_animation()`: Cho các thuật toán tự tạo câu đố.
    *   **Thực thi Thuật toán và Xử lý Kết quả**:
        *   `run_algorithm_threaded()`: Khởi tạo một luồng mới để chạy thuật toán đã chọn.
        *   `_solver_thread_func()`: Hàm mục tiêu cho luồng, gọi phương thức thuật toán tương ứng trên đối tượng `Puzzle`.
        *   `_check_result_queue()`: Kiểm tra hàng đợi để lấy kết quả từ luồng thuật toán.
        *   `_process_solver_result()`: Xử lý kết quả trả về từ thuật toán (cập nhật GUI, bắt đầu trực quan hóa).
        *   `_update_visualization()`: Điều khiển hoạt ảnh từng bước của đường đi giải pháp.
        *   `run_benchmark_threaded()`, `_benchmark_thread_func()`, `_process_benchmark_result()`: Logic cho chế độ benchmark.
    *   `update()`: Cập nhật logic của ứng dụng mỗi khung hình.
    *   `draw()`: Vẽ lại toàn bộ màn hình mỗi khung hình.
    *   `run()`: Vòng lặp chính của ứng dụng.

7.  **Khối Thực thi Chính (`if __name__ == "__main__":`)**:
    *   Tạo một instance của `PygamePuzzleApp` và gọi phương thức `run()` của nó để khởi động ứng dụng.
    *   Bao gồm xử lý ngoại lệ cơ bản.
  
## GIF ghi lại cách hoạt động của các thuật toán
  * CÁC THUẬT TOÁN THUỘC NHÓM TÌM KIẾM KHÔNG CÓ THÔNG TIN
    BFS:
    
    ![Thuat-toan-bfs](https://github.com/user-attachments/assets/f0686795-81f1-4974-b215-c6bb568acbf4)
    
    DFS:
    
    ![dfs-_online-video-cutter com_](https://github.com/user-attachments/assets/b3029776-ec84-4b6f-b350-26bd7d4760ff)
    
    IDDFS:
    
    ![iddfs (online-video-cutter com)](https://github.com/user-attachments/assets/d293a22f-f515-4107-a8f9-76502a4b1750)
    
    UCS:
    
    ![ucs (online-video-cutter com)](https://github.com/user-attachments/assets/0335d5e8-e84a-4cfa-a5ae-2db1efdc2354)
    
    
  * CÁC THUẬT TOÁN THUỘC NHÓM TÌM KIẾM CÓ THÔNG TIN
  * 
    A*:
    ![a_star-_online-video-cutter com_](https://github.com/user-attachments/assets/0dc8ec7b-9c84-4b07-bc8f-eeaa11c94940)
    
    IDA*:
    
    ![ida_star (online-video-cutter com)](https://github.com/user-attachments/assets/89aad33d-c960-48bd-94c4-2ba5d5ee6415)

      * CÁC THUẬT TOÁN LOCAL SEARCH
  
    Beam Search:
    ![beam-_online-video-cutter com_](https://github.com/user-attachments/assets/381f5825-0739-439d-9262-713da9ead1a4)

    Simulated Annealing:
    ![sa (online-video-cutter com)](https://github.com/user-attachments/assets/2f3f901f-1201-42ab-9d72-dde8a63542e2)

    Genetic Algorithm:
    ![genetic (online-video-cutter com)](https://github.com/user-attachments/assets/61ea98e5-9861-4d49-99db-3633c85c4456)

    Simple hill climbing:
    ![spl (online-video-cutter com)](https://github.com/user-attachments/assets/6848de4c-b1b8-4873-a3eb-17f315850694)

    Steepest-ascent hill climbing
    ![stee (online-video-cutter com)](https://github.com/user-attachments/assets/c3a794c0-26a3-4e89-a491-55b4d954e0e1)

    Stochastic hill climbing
    ![stouch (online-video-cutter com)](https://github.com/user-attachments/assets/a7b0a6ef-1197-4f14-8dc2-b056a3ad0948)

  * CÁC THUẬT TOÁN TÌM KIẾM TRONG MÔI TRƯỜNG PHỨC TẠP
    
    AND OR SEARCH:
    ![abd_or-_online-video-cutter com_](https://github.com/user-attachments/assets/1236d56d-3e1a-4e7a-9f8f-ec2993488a9a)

    BELIEF STATE SEARCH:
    
    
      * CÁC THUẬT TOÁN THUỘC NHÓM CSPs
        
    BACKTRACKING:
    ![backtracking-_online-video-cutter com_](https://github.com/user-attachments/assets/765f684b-2aba-4214-b077-55226e57ea2d)

    BACKTRACKING WITH FORWARD CHECKING:
    ![backtrack_up-_2_-_online-video-cutter com_](https://github.com/user-attachments/assets/21242255-3601-4c3b-9321-6f5ba5c68f4e)

   * CÁC THUẬT TOÁN THUỘC NHÓM REINFORCEMENT LEARNING
     
     Q-LEARING
    ![Q (online-video-cutter com)](https://github.com/user-attachments/assets/82e123d5-7d01-4f1e-8838-cf30740ee1c8)

## Những Cải Tiến Tiềm Năng

*   Thêm nhiều heuristic khác (ví dụ: Linear Conflict).
*   Cho phép người dùng chọn heuristic để sử dụng.
*   Lưu và tải trạng thái câu đố.
*   Cải thiện hiệu suất của một số thuật toán.
*   Thêm tùy chọn để tạm dừng/tiếp tục/bỏ qua trực quan hóa.
*   Làm cho giao diện người dùng đáp ứng tốt hơn với các kích thước cửa sổ khác nhau.
*   Thêm nhiều thông tin thống kê chi tiết hơn cho mỗi thuật toán.
*   Lưu kết quả benchmark vào file.
