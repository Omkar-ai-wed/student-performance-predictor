import { useState } from 'react'

function App() {
  const [formData, setFormData] = useState({
    attendance_rate: '',
    weekly_study_hours: '',
    past_exam_scores: '',
    homework_completion_rate: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const payload = {
        attendance_rate: parseFloat(formData.attendance_rate),
        weekly_study_hours: parseFloat(formData.weekly_study_hours),
        past_exam_scores: parseFloat(formData.past_exam_scores),
        ...(formData.homework_completion_rate && {
          homework_completion_rate: parseFloat(formData.homework_completion_rate)
        })
      }

      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Failed to get prediction')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-teal-50 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left Column: Context / Intro */}
        <div className="flex flex-col justify-center space-y-6">
          <h1 className="text-4xl lg:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-teal-500 tracking-tight">
            Predict Student Success
          </h1>
          <p className="text-lg text-gray-600 leading-relaxed">
            Gain AI-driven insights into potential student outcomes based on their historical performance and engagement metrics.
          </p>
          <div className="space-y-4 pt-4">
            <div className="flex items-center space-x-3 text-sm text-gray-500 bg-white/50 p-3 rounded-lg border border-white/40">
              <svg className="w-6 h-6 text-indigo-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <span>Utilises advanced Gradient Boosting Regressor</span>
            </div>
            <div className="flex items-center space-x-3 text-sm text-gray-500 bg-white/50 p-3 rounded-lg border border-white/40">
              <svg className="w-6 h-6 text-teal-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
              <span>Instant evaluation with median imputation</span>
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Form */}
        <div className="glass-panel rounded-2xl p-8 relative overflow-hidden transition-all duration-300 hover:shadow-2xl">
          <div className="absolute top-0 right-0 -mt-10 -mr-10 w-40 h-40 bg-indigo-500 opacity-10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-0 -mb-10 -ml-10 w-40 h-40 bg-teal-500 opacity-10 rounded-full blur-3xl"></div>
          
          <form onSubmit={handleSubmit} className="relative z-10 space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Attendance Rate (%)</label>
              <input 
                type="number" 
                name="attendance_rate"
                required 
                min="0" max="100" step="0.1"
                value={formData.attendance_rate} 
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white/80 transition-all text-gray-800"
                placeholder="e.g. 85.5"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Weekly Study Hours</label>
              <input 
                type="number" 
                name="weekly_study_hours"
                required 
                min="0" max="168" step="0.5"
                value={formData.weekly_study_hours} 
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white/80 transition-all text-gray-800"
                placeholder="e.g. 12"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Past Exam Scores (Average 0-100)</label>
              <input 
                type="number" 
                name="past_exam_scores"
                required 
                min="0" max="100" step="0.1"
                value={formData.past_exam_scores} 
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white/80 transition-all text-gray-800"
                placeholder="e.g. 74.5"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Homework Completion Rate (%) <span className="text-gray-400 font-normal">(Optional)</span></label>
              <input 
                type="number" 
                name="homework_completion_rate"
                min="0" max="100" step="0.1"
                value={formData.homework_completion_rate} 
                onChange={handleChange}
                className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white/80 transition-all text-gray-800 placeholder-gray-300"
                placeholder="Leave blank to use median interpolation"
              />
            </div>

            <button 
              type="submit" 
              disabled={loading}
              className={`w-full py-3.5 px-4 bg-gradient-to-r from-indigo-600 to-teal-500 hover:from-indigo-700 hover:to-teal-600 text-white font-semibold rounded-xl transition-all shadow-md hover:shadow-lg flex justify-center items-center ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
            >
              {loading ? (
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              ) : (
                'Generate Predictor'
              )}
            </button>
          </form>

          {/* Result / Error Display */}
          <div className="mt-6 relative z-10 lg:min-h-[140px]">
            {error && (
              <div className="p-4 bg-red-50 text-red-600 rounded-xl border border-red-100 text-sm animate-[fadeIn_0.3s_ease-out]">
                <div className="font-semibold mb-1 flex items-center">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  Prediction Error
                </div>
                {error}
              </div>
            )}
            
            {result && !error && (
              <div className="p-5 bg-gradient-to-br from-indigo-50 to-white rounded-xl border border-indigo-100 shadow-inner animate-[fadeIn_0.4s_ease-out]">
                <div className="text-sm font-medium text-indigo-400 mb-1 uppercase tracking-wider">Estimated Final Grade</div>
                <div className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-teal-500 mb-3">
                  {result.predicted_grade}<span className="text-2xl text-indigo-300 ml-1">%</span>
                </div>
                <div className="text-xs text-gray-500 bg-white/60 p-2 rounded leading-snug">
                  {result.confidence_note}
                </div>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}

export default App
