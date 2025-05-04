# Project-AI
Đây là một ứng dụng Python giải bài toán 8-Puzzle (Câu đố 8 số) cổ điển bằng cách sử dụng nhiều thuật toán tìm kiếm Trí tuệ Nhân tạo (AI). Ứng dụng có giao diện người dùng đồ họa (GUI) được xây dựng bằng Pygame để người dùng tương tác, trực quan hóa quá trình giải và đánh giá hiệu năng các thuật toán khác nhau.

Tính Năng:
    Bảng Câu Đố 8 Số Tương Tác: Đặt trạng thái ban đầu bằng cách nhấp vào các ô hoặc tạo ngẫu nhiên một bảng có thể giải được.
    Nhiều Thuật Toán AI: Triển khai một loạt các thuật toán tìm kiếm (xem danh sách bên dưới).
    Trực Quan Hóa: Xem câu đố được giải từng bước (hoặc quá trình khám phá đối với một số thuật toán).
    Hiển Thị Kết Quả: Hiển thị đường đi giải pháp (các nước đi), số bước/trạng thái đã khám phá và thời gian thực hiện.
    Đánh Giá Thuật Toán (Benchmark): So sánh hiệu năng (thời gian, số bước) của các thuật toán tìm đường khác nhau trên câu đố hiện tại.
    Kiểm Tra Khả Năng Giải: Tự động xác minh xem bảng do người dùng nhập có thể giải được hay không.
    Tham Số Điều Chỉnh Được: Điều khiển tốc độ trực quan hóa và độ sâu tối đa cho các tìm kiếm giới hạn độ sâu thông qua thanh trượt.
    Giao Diện Người Dùng Thông Tin: Bao gồm các thông báo về trạng thái hiện tại và các ghi chú hữu ích.

Yêu Cầu
    Python 3.x
    Thư viện Pygame
    
Cài Đặt
    Đảm bảo đã cài đặt Python 3.
    Cài đặt Pygame: Mở terminal hoặc command prompt của bạn và chạy:
        pip install pygame

Cách Sử Dụng
    Chạy script:   python ten_file_cua_ban.py
        (Thay ten_file_cua_ban.py bằng tên thực tế của file Python của bạn).
    Đặt Trạng Thái Ban Đầu:
        Nhập Thủ Công: Nhấp lần lượt vào các ô trống trên lưới để đặt các số từ 1 đến 8. Ứng dụng sẽ tự động điền số 0 vào ô cuối cùng. Nó sẽ kiểm tra khả năng giải sau khi tất cả các số được đặt.
        Tạo Ngẫu Nhiên: Nhấp vào các nút như Backtrack hoặc CSP Solve. Các thuật toán này trước tiên sẽ tạo ra một bảng ngẫu nhiên có thể giải được (được hiển thị trực quan), sau đó cố gắng giải nó, hiển thị đường đi khám phá/giải pháp. Genetic cũng bắt đầu với trạng thái ngẫu nhiên nhưng hiển thị trực tiếp trạng thái tốt nhất tìm thấy sau các thế hệ (không trực quan hóa đường đi).
        Chọn Thuật Toán: Khi trạng thái bảng hợp lệ và có thể giải được (thông báo "Board ready!"), hãy nhấp vào một trong các nút thuật toán (ví dụ: BFS, A*, Simple HC) để bắt đầu quá trình giải.
Trực Quan Hóa: Nếu thuật toán được chọn tạo ra một đường đi hoặc chuỗi khám phá, các bước sẽ được hoạt họa trên lưới. Sử dụng thanh trượt "Vis Speed(ms)" để điều khiển độ trễ giữa các bước.
        Xem Kết Quả: Bảng điều khiển bên phải hiển thị thông tin về quá trình thực thi của thuật toán, bao gồm trạng thái (đã giải, thất bại, hoàn thành), thời gian thực hiện, số bước/trạng thái và chuỗi các nước đi đối với các thuật toán tìm đường đã giải được. Sử dụng con lăn chuột để cuộn qua các kết quả dài.
        Benchmark (Đánh giá): Sau khi đặt một bảng có thể giải được, nhấp vào nút Benchmark để chạy một số thuật toán tìm đường và so sánh hiệu suất của chúng trong bảng kết quả.
        Reset (Đặt lại): Nhấp vào nút Reset bất kỳ lúc nào để xóa bảng, kết quả và quay lại trạng thái nhập ban đầu.
Điều Chỉnh Tham Số:
Vis Speed(ms): Điều khiển tốc độ hoạt ảnh (giá trị thấp hơn = nhanh hơn).
Max Depth: Đặt giới hạn độ sâu cho các thuật toán như IDDFS, DLS Visit (And/Or Search) và ảnh hưởng đến việc cắt tỉa của DFS.

Các Thuật Toán Được Triển Khai
    Tìm Kiếm Mù (Uninformed Search):
        Tìm Kiếm Theo Chiều Rộng (BFS)
        Tìm Kiếm Chi Phí Đồng Nhất (UCS)
        Tìm Kiếm Theo Chiều Sâu (DFS) (có cắt tỉa theo độ sâu)
        Tìm Kiếm Sâu Dần Lặp (IDDFS)
    Tìm Kiếm Có Thông Tin (Heuristic - Khoảng cách Manhattan):
        Tìm Kiếm Tham Lam Tốt Nhất Đầu Tiên (Greedy Best-First Search)
        Tìm Kiếm A* (A* Search)
        Tìm Kiếm A* Sâu Dần Lặp (IDA*)
        Tìm Kiếm Chùm (Beam Search)
    Tìm Kiếm Cục Bộ & Tối Ưu Hóa:
        Leo Đồi Đơn Giản (Simple Hill Climbing - First Choice)
        Leo Đồi Dốc Nhất (Steepest Ascent Hill Climbing)
        Leo Đồi Ngẫu Nhiên (Stochastic Hill Climbing)
        Luyện Nhiệt Mô Phỏng (Simulated Annealing - SA)
        Giải Thuật Di Truyền (Genetic Algorithm - GA)
    Thỏa Mãn Ràng Buộc / Khám Phá:
        Quay Lui (Backtracking - Tạo bảng ngẫu nhiên, sau đó giải/khám phá từ đó)
        Giải CSP (CSP Solve - Được triển khai tương tự Backtracking cho bài toán này)
        Duyệt DLS (DLS Visit - Tìm kiếm giới hạn độ sâu, được sử dụng như một khái niệm tương đương/thay thế cho tìm kiếm đồ thị And/Or trong ngữ cảnh này)

Cấu Trúc Code
    Lớp Puzzle: Chứa logic cốt lõi cho trạng thái câu đố 8 số, kiểm tra mục tiêu, tạo trạng thái lân cận, tính toán heuristic và triển khai tất cả các thuật toán tìm kiếm.
    Lớp PygamePuzzleApp: Quản lý GUI Pygame, xử lý đầu vào của người dùng (nhấp chuột, cuộn), kiểm soát trạng thái ứng dụng (nhập, đang chạy, trực quan hóa, v.v.), quản lý thực thi thuật toán thông qua luồng (threading), hiển thị lưới và kết quả, xử lý hoạt ảnh trực quan hóa và đánh giá hiệu năng.
    Hàm Hỗ Trợ (Helper Functions): Các hàm tiện ích khác nhau để chuyển đổi trạng thái (state_to_tuple), kiểm tra tính hợp lệ (is_valid_state), kiểm tra khả năng giải (is_solvable, get_inversions), tạo trạng thái ngẫu nhiên (generate_random_solvable_state) và trích xuất các nước đi từ một đường dẫn (get_solution_moves).
    Hằng Số (Constants): Định nghĩa màu sắc, kích thước màn hình, cài đặt phông chữ, giới hạn thuật toán (ví dụ: MAX_BFS_STATES) và trạng thái mục tiêu.
