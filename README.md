To include `curl` examples in your `README.md` for interacting with the API endpoints, you can add sections that demonstrate how to make `curl` requests to the available endpoints. Here's the updated `README.md` with `curl` examples:

---

# FastAPI Multilingual Embeddings API

This project provides a FastAPI service for generating and comparing multilingual text embeddings using the Hugging Face `intfloat/multilingual-e5-large` model. The service includes endpoints for obtaining similarity scores and raw embeddings.

## Project Structure

- `app/`
  - `main.py`: Contains the FastAPI application and API endpoints.
  - `services.py`: Contains the logic for processing embeddings and calculating similarity scores.
  - `models.py`: Contains Pydantic models for request data validation.
- `tests/`
  - `test_main.py`: Contains test cases for the FastAPI endpoints.
- `start.sh`: A script to set up and run the application.
- `requirements.txt`: Lists project dependencies.

## Prerequisites

- Python 3.7 or higher
- Conda with the environment named `huggingface`
- NVIDIA GPU (for GPU acceleration, ensure CUDA and the NVIDIA Container Toolkit are installed)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create and Activate Conda Environment**

   ```bash
   conda create --name huggingface python=3.8
   conda activate huggingface
   ```

3. **Install Dependencies**

   Create or update the `requirements.txt` file with the latest versions:

   ```plaintext
   fastapi>=0.95.0
   uvicorn>=0.23.0
   transformers>=4.32.0
   torch>=2.1.0+cu118
   pydantic>=2.2.0
   pytest>=7.4.0
   httpx>=0.24.0
   pytest-sugar
   ```

   Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the FastAPI Server**

   Use the `start.sh` script to set up the environment and run the server:

   ```bash
   ./start.sh
   ```

   The application will be available at `http://localhost:8000`.

2. **Access the API Documentation**

   Open your web browser and navigate to:

   - **Swagger UI**: `http://localhost:8000/docs`
   - **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### 1. Get Similarity Scores

- **Endpoint**: `POST /score-embedding`
- **Description**: Compute similarity scores between query and passage embeddings.

- **Request Body Example**:

  ```json
  {
    "texts": [
      "query: how much protein should a female eat",
      "query: Công thức nấu ăn bí ngô tự làm",
      "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
      "passage: 1. Bí ngô xào sợi Nguyên liệu: nửa quả bí ngô mềm Gia vị: hành, muối, đường, cốt gà Cách làm: 1. Dùng dao gọt bỏ một lớp vỏ mỏng trên bề mặt bí ngô, dùng dao cạo sạch phần thịt thìa 2. Bào thành từng sợi mỏng (Nếu không có thớt thì dùng dao cắt từ từ thành từng sợi mỏng) 3. Đun nóng nồi, cho dầu vào, cho hành lá cắt nhỏ vào xào cho đến khi có mùi thơm 4. Thêm Bí ngô cắt nhỏ và xào nhanh trong khoảng một phút, thêm muối, một ít đường và nước cốt gà cho vừa ăn rồi dùng 2. Bí ngô xào hẹ Nguyên liệu: 1 quả bí ngô Gia vị: hẹ, tỏi băm, dầu ô liu, muối Cách làm: 1. Gọt vỏ bí ngô và cắt thành từng lát 2. Sau khi chảo dầu nóng 80%, cho tỏi băm vào xào cho đến khi có mùi thơm 3. Sau khi xào xong, cho các lát bí đỏ vào xào chín 4. Trong khi xào, bạn. Thỉnh thoảng có thể thêm nước vào nồi nhưng không quá nhiều 5. Thêm muối và xào đều 6. Bí đỏ gần mềm sau đó có thể tắt lửa 7. Rắc vào. hẹ và phục vụ."
    ]
  }
  ```

- **Response Example**:

  ```json
  {
    "scores": [[score1, score2], [score3, score4]]
  }
  ```

- **`curl` Example**:

  ```bash
  curl -X POST "http://localhost:8000/score-embedding" -H "Content-Type: application/json" -d '{
    "texts": [
      "query: how much protein should a female eat",
      "query: Công thức nấu ăn bí ngô tự làm",
      "passage: As a general guideline, the CDC\'s average requirement of protein for women ages 19 to 70 is 46 grams per day.",
      "passage: 1. Bí ngô xào sợi Nguyên liệu: nửa quả bí ngô mềm Gia vị: hành, muối, đường, cốt gà Cách làm: 1. Dùng dao gọt bỏ một lớp vỏ mỏng trên bề mặt bí ngô, dùng dao cạo sạch phần thịt thìa 2. Bào thành từng sợi mỏng (Nếu không có thớt thì dùng dao cắt từ từ thành từng sợi mỏng) 3. Đun nóng nồi, cho dầu vào, cho hành lá cắt nhỏ vào xào cho đến khi có mùi thơm 4. Thêm Bí ngô cắt nhỏ và xào nhanh trong khoảng một phút, thêm muối, một ít đường và nước cốt gà cho vừa ăn rồi dùng 2. Bí ngô xào hẹ Nguyên liệu: 1 quả bí ngô Gia vị: hẹ, tỏi băm, dầu ô liu, muối Cách làm: 1. Gọt vỏ bí ngô và cắt thành từng lát 2. Sau khi chảo dầu nóng 80%, cho tỏi băm vào xào cho đến khi có mùi thơm 3. Sau khi xào xong, cho các lát bí đỏ vào xào chín 4. Trong khi xào, bạn. Thỉnh thoảng có thể thêm nước vào nồi nhưng không quá nhiều 5. Thêm muối và xào đều 6. Bí đỏ gần mềm sau đó có thể tắt lửa 7. Rắc vào. hẹ và phục vụ."
    ]
  }'
  ```

### 2. Get Raw Embeddings

- **Endpoint**: `POST /text-embedding`
- **Description**: Get raw embeddings for the provided texts.

- **Request Body Example**:

  ```json
  {
    "texts": [
      "query: how much protein should a female eat",
      "query: Công thức nấu ăn bí ngô tự làm"
    ]
  }
  ```

- **Response Example**:

  ```json
  {
    "embeddings": [[embedding1], [embedding2]]
  }
  ```

- **`curl` Example**:

  ```bash
  curl -X POST "http://localhost:8000/text-embedding" -H "Content-Type: application/json" -d '{
    "texts": [
      "query: how much protein should a female eat",
      "query: Công thức nấu ăn bí ngô tự làm"
    ]
  }'
  ```

## Running Tests

1. **Install Test Dependencies**

   Ensure `pytest`, `pytest-sugar`, and `httpx` are installed:

   ```bash
   pip install pytest pytest-sugar httpx
   ```

2. **Run the Tests**

   Execute the tests with:

   ```bash
   pytest
   ```

   The output will be enhanced with color and progress bars, thanks to `pytest-sugar`.

## Docker Setup (Optional)

### Build and Run with Docker Compose

1. **Ensure Docker and NVIDIA Docker Support are Installed**

2. **Build and Start the Docker Containers**

   ```bash
   docker-compose up --build
   ```

   The service will be available at `http://localhost:8000`.

3. **Stop the Containers**

   ```bash
   docker-compose down
   ```

## Notes

- **GPU Support**: Ensure you have the necessary NVIDIA drivers and CUDA installed for GPU acceleration. Modify the `Dockerfile` or `requirements.txt` for specific versions if needed.
- **Environment Variables**: For production setups, consider using environment variables or `.env` files to manage sensitive configurations.

## Contributing

Feel free to contribute by opening issues or submitting pull requests. For detailed contribution guidelines, please refer to the project's CONTRIBUTING.md file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

This updated `README.md` includes `curl` examples for making requests to the API endpoints, which should help users interact with your FastAPI service more easily. Adjust the example data and URLs as needed based on your actual setup.