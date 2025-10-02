# 🌟 Keratrack - AI-Powered Hair Health Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)

Keratrack is a comprehensive AI-powered platform that helps users track, analyze, and improve their hair health through advanced machine learning algorithms and personalized nutrition recommendations.

## ✨ Features

### 🔬 Core Hair Analysis
- **AI-Powered Hair Loss Detection**: Uses ConvNeXt deep learning model to classify hair loss stages (LEVEL_2 to LEVEL_7)
- **Confidence Scoring**: Provides accuracy percentage for each analysis
- **Progress Tracking**: Monitor hair health changes over time
- **Image Processing**: Supports multiple hair view angles (Front, Top-Down, Side)

### 🥗 AI Diet Recommendations
- **Personalized Nutrition Plans**: AI-generated meal plans based on hair loss stage and health profile
- **Smart Nutrient Targeting**: Calculates optimal daily requirements for hair-healthy nutrients
- **Supplement Recommendations**: Evidence-based supplement suggestions with dosage and timing
- **Dietary Restriction Support**: Accommodates vegetarian, vegan, gluten-free, and other dietary needs

### 📊 Lifestyle Tracking
- **Daily Health Metrics**: Track stress levels, sleep hours, exercise, and water intake
- **Correlation Analysis**: Understand how lifestyle factors affect hair health
- **Progress Analytics**: Visual charts and insights on your health journey
- **Goal Setting**: Set and monitor hair health improvement goals

### 🔐 User Management
- **Secure Authentication**: JWT-based user authentication system
- **Personal Dashboard**: User-specific data and recommendations
- **Privacy Protection**: Secure storage of sensitive health data

## 🏗️ Technical Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.8+
- **Database**: PostgreSQL with SQLAlchemy ORM
- **AI/ML**: PyTorch + Timm for deep learning models
- **Authentication**: JWT tokens with bcrypt password hashing
- **API Documentation**: Automatic OpenAPI/Swagger documentation

### Frontend (Next.js)
- **Framework**: Next.js 15.5 with TypeScript
- **Styling**: Tailwind CSS for responsive design
- **Charts**: Chart.js for data visualization
- **State Management**: React hooks and local storage
- **UI Components**: Custom responsive components

### Database
- **Primary DB**: PostgreSQL (Neon Cloud)
- **ORM**: SQLAlchemy with Alembic migrations
- **Tables**: Users, Predictions, Diet Assessments, Recommendations, Lifestyle Entries

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8 or higher (I used 3.12)**
- **Node.js 18 or higher (I used 20.17.0)**
- **Git**
- **PostgreSQL** (or access to a cloud database like Neon)

### 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Paresh-0007/Keratrack.git
   cd Keratrack
   ```

### 🔧 Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```
2. **Create and activate virtual environment**
   ```bash
   # Windows
   pip install virtualenv
   python -m venv keratrackvenv
   keratrackvenv\Scripts\activate

   # macOS/Linux
   pip install virtualenv
   python3 -m venv keratrackvenv
   source keratrackvenv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you encounter with ModuleNotFoundError for any dependencies, run:
   ```bash
   pip install <missing-module-name>
   ```
   then please update the `requirements.txt` file by running:
   ```bash
   pip freeze > requirements.txt
   ```

4. **Environment Configuration**
   
   Create a `.env` file in the backend directory:
   ```env
   # Database Configuration
   POSTGRES_URL=postgresql://username:password@host:port/database_name
   
   # JWT Configuration
   SECRET_KEY=your-secret-key-here-please-change-in-production
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

   **🔑 Database Setup Options:**

   **Option A: Use Neon (Recommended)**
   - Sign up at [neon.tech](https://neon.tech)
   - Create a new project
   - Copy the connection string to your `.env` file

   **Option B: Local PostgreSQL**
   ```bash
   # Install PostgreSQL locally and create database
   createdb keratrack
   # Update POSTGRES_URL in .env with local connection string
   POSTGRES_URL=postgresql://localhost/keratrack
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Download AI Model**
   
   Ensure you have the hair loss classification model file:
   - Place `best_convnext_hairfall.pth` in the backend directory
   - This file should be included in the repository or available from the model training process

7. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Redoc Documentation: `http://localhost:8000/redoc`

### 🎨 Frontend Setup

1. **Navigate to frontend directory** (in a new terminal)
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:3000`

## 🧪 Testing the Application

### 1. **Hair Analysis Feature**
- Go to `http://localhost:3000`
- Upload a hair image (front, top, or side view)
- Receive AI-powered hair loss stage classification

### 2. **Diet Recommendations**
- Navigate to `http://localhost:3000/diet`
- Complete the comprehensive diet assessment
- Receive personalized nutrition recommendations

### 3. **Lifestyle Tracking**
- Visit `http://localhost:3000/diet/lifestyle`
- Log daily health metrics (stress, sleep, exercise, water)
- View correlations with hair health progress

## 📁 Project Structure

```
Keratrack/
├── backend/
│   ├── app/
│   │   ├── auth.py              # Authentication logic
│   │   ├── crud.py              # Database operations
│   │   ├── database.py          # Database configuration
│   │   ├── diet_ai.py           # AI diet recommendation engine
│   │   ├── main.py              # FastAPI application and routes
│   │   ├── ml_interface.py      # Hair analysis ML model interface
│   │   ├── models.py            # SQLAlchemy database models
│   │   └── schemas.py           # Pydantic schemas for API
│   ├── alembic/                 # Database migrations
│   ├── uploads/                 # Uploaded hair images
│   ├── best_convnext_hairfall.pth  # AI model file
│   ├── requirements.txt         # Python dependencies
│   └── .env                     # Environment variables
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── diet/
│   │   │   │   ├── page.tsx     # Diet recommendations dashboard
│   │   │   │   └── lifestyle/
│   │   │   │       └── page.tsx # Lifestyle tracking interface
│   │   │   ├── page.tsx         # Main landing page
│   │   │   └── ...              # Other pages
│   │   └── ...
│   ├── package.json             # Node.js dependencies
│   └── ...
└── README.md                    # This file
```

## 🔑 API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /token` - User login

### Hair Analysis
- `POST /predict` - Upload and analyze hair image
- `GET /history` - Get user's analysis history

### Diet Recommendations
- `POST /diet/assessment` - Create diet assessment
- `GET /diet/recommendations` - Get AI-powered diet recommendations
- `POST /diet/lifestyle` - Log lifestyle entry
- `GET /diet/lifestyle/history` - Get lifestyle tracking history
- `POST /diet/food-log` - Log food intake
- `GET /diet/progress-analysis` - Get diet effectiveness analysis

## 🤖 AI Models

### Hair Loss Classification Model
- **Architecture**: ConvNeXt Base
- **Classes**: 6 hair loss stages (LEVEL_2 to LEVEL_7)
- **Input**: 224x224 RGB images
- **Output**: Classification with confidence score

### Diet Recommendation Engine
- **Type**: Rule-based AI with statistical analysis
- **Features**: 
  - Personalized nutrient calculation
  - Hair stage correlation
  - Lifestyle factor integration
  - Meal plan generation

## 🔧 Development

### Adding New Features

1. **Backend Changes**:
   - Add new models in `app/models.py`
   - Create database migration: `alembic revision --autogenerate -m "description"`
   - Add API endpoints in `app/main.py`
   - Update schemas in `app/schemas.py`

2. **Frontend Changes**:
   - Create new pages in `src/app/`
   - Add components for new features
   - Update navigation and routing

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new feature"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🌐 Deployment

### Backend Deployment (Suggested: Railway/Render)

1. **Environment Variables**:
   ```env
   POSTGRES_URL=your-production-database-url
   SECRET_KEY=your-production-secret-key
   ```

2. **Deploy Command**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

### Frontend Deployment (Suggested: Vercel/Netlify)

1. **Build Command**: `npm run build`
2. **Start Command**: `npm start`
3. **Environment Variables**: Update API endpoints for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Python PEP 8 style guide for backend code
- Use TypeScript and ESLint rules for frontend code
- Write descriptive commit messages
- Add tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Verify your `POSTGRES_URL` in `.env`
   - Check if database service is running
   - Ensure database exists and is accessible

2. **Model File Not Found**
   - Ensure `best_convnext_hairfall.pth` is in the backend directory
   - Check file permissions and size

3. **Frontend API Connection Issues**
   - Verify backend is running on `http://localhost:8000`
   - Check for CORS configuration in backend
   - Ensure JWT tokens are being sent correctly

4. **Migration Errors**
   - Clear migration files and regenerate if needed
   - Check database permissions
   - Verify model definitions are correct

### Getting Help

- Check the [Issues](https://github.com/Paresh-0007/Keratrack/issues) page for known problems
- Create a new issue with detailed description and error logs
- Join our community discussions

## 🎯 Roadmap

- [ ] Mobile app development (React Native)
- [ ] Advanced ML models for scalp health analysis
- [ ] Integration with wearable devices
- [ ] Telemedicine platform integration
- [ ] Multi-language support
- [ ] Real-time notifications system
- [ ] Advanced analytics dashboard

## 👥 Team

- **Paresh Gupta** - [@Paresh-0007](https://github.com/Paresh-0007) - Project Lead & Full Stack Developer

## 🙏 Acknowledgments

- ConvNeXt model architecture by Facebook Research
- Timm library for model implementations
- FastAPI and Next.js communities for excellent frameworks
- Hair loss research community for domain knowledge

---

**Made with ❤️ for better hair health worldwide**