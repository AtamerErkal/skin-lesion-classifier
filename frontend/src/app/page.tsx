"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster, toast } from "sonner";
import { 
  Camera, 
  Upload, 
  ScanLine, 
  FileText, 
  Shield, 
  Activity,
  AlertTriangle,
  CheckCircle2,
  X,
  ChevronRight,
  Microscope,
  Stethoscope,
  Clock,
  Info,
  Sparkles,
  Brain,
  Scan,
  ArrowRight,
  History,
  Maximize2,
  Loader2
} from "lucide-react";

// Animation variants
const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const scaleIn = {
  initial: { scale: 0.9, opacity: 0 },
  animate: { scale: 1, opacity: 1 },
  exit: { scale: 0.9, opacity: 0 }
};

interface Prediction {
  predicted_class: string;
  predicted_class_full: string;
  confidence: number;
  risk_level: string;
  all_probabilities: Array<{ class_name: string; probability: number }>;
  grad_cam_image: string;
}

const lesionInfo = {
  'nv': {
    name: 'Melanocytic Nevi',
    fullName: 'Melanocytic nevi (Mole)',
    risk: 'LOW',
    description: 'Common moles, typically harmless pigmented skin lesions. Most common type (~67% of cases).',
    recommendation: 'Regular monitoring, annual dermatology check-ups',
    abcd: 'Usually symmetrical, uniform color, sharp borders',
    color: '#059669'
  },
  'mel': {
    name: 'Melanoma',
    fullName: 'Melanoma (Malignant)',
    risk: 'HIGH',
    description: 'Most dangerous form of skin cancer. Life-threatening if not detected early.',
    recommendation: 'URGENT dermatologist consultation required',
    abcd: 'Often asymmetrical, irregular borders, color variation',
    color: '#dc2626'
  },
  'bkl': {
    name: 'Benign Keratosis',
    fullName: 'Benign keratosis-like lesions',
    risk: 'MEDIUM',
    description: 'Seborrheic keratosis, solar lentigo, lichen planus. Non-cancerous growths.',
    recommendation: 'Monitor for changes, professional evaluation advised',
    abcd: 'Usually well-defined, uniform appearance',
    color: '#d97706'
  },
  'bcc': {
    name: 'Basal Cell Carcinoma',
    fullName: 'Basal cell carcinoma',
    risk: 'HIGH',
    description: 'Most common form of skin cancer, slow-growing. Treatable with early detection.',
    recommendation: 'Dermatologist appointment within 1-2 weeks',
    abcd: 'Pearly appearance, may have visible blood vessels',
    color: '#dc2626'
  },
  'akiec': {
    name: 'Actinic Keratoses',
    fullName: 'Actinic keratoses',
    risk: 'HIGH',
    description: 'Precancerous lesions caused by sun exposure. Can progress to squamous cell carcinoma.',
    recommendation: 'Professional evaluation within one month',
    abcd: 'Rough, scaly patches, often sun-damaged skin',
    color: '#dc2626'
  },
  'vasc': {
    name: 'Vascular Lesions',
    fullName: 'Vascular lesions',
    risk: 'LOW',
    description: 'Angiomas, angiokeratomas, pyogenic granulomas. Blood vessel growths, usually harmless.',
    recommendation: 'Routine monitoring',
    abcd: 'Red/purple, may blanch with pressure',
    color: '#059669'
  },
  'df': {
    name: 'Dermatofibroma',
    fullName: 'Dermatofibroma',
    risk: 'LOW',
    description: 'Fibrous tissue growth in the skin. Harmless, firm nodules.',
    recommendation: 'No treatment needed unless symptomatic',
    abcd: 'Firm nodules, may dimple when pinched',
    color: '#059669'
  }
};

// Healthcare color palette
const healthcareColors = {
  primary: '#0f766e',      // Teal 700 - main brand
  primaryLight: '#14b8a6', // Teal 500
  primaryDark: '#0d9488',  // Teal 600
  secondary: '#0284c7',    // Sky 600
  accent: '#06b6d4',       // Cyan 500
  success: '#059669',      // Emerald 600
  warning: '#d97706',      // Amber 600
  danger: '#dc2626',       // Red 600
  background: '#f8fafc',   // Slate 50
  card: '#ffffff',
  text: '#1e293b',         // Slate 800
  textMuted: '#64748b',    // Slate 500
  border: '#e2e8f0',       // Slate 200
};

// Particle Background Component - Client Only to avoid hydration mismatch
const ParticleBackground = () => {
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  if (!mounted) return null;
  
  // Generate deterministic positions based on index
  const particles = [...Array(20)].map((_, i) => ({
    x: (i * 137.5) % (typeof window !== 'undefined' ? window.innerWidth : 1000),
    y: (i * 89.7) % (typeof window !== 'undefined' ? window.innerHeight : 800),
    duration: 10 + (i % 10),
    delay: (i % 5) * 0.5,
  }));
  
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle, i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 bg-teal-500/20 rounded-full"
          initial={{
            x: particle.x,
            y: particle.y,
          }}
          animate={{
            y: particle.y - 100,
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            delay: particle.delay,
          }}
        />
      ))}
    </div>
  );
};

// Step Indicator Component
const StepIndicator = ({ currentStep }: { currentStep: number }) => {
  const steps = [
    { icon: Upload, label: "Upload" },
    { icon: Scan, label: "Scan" },
    { icon: Brain, label: "Analyze" },
    { icon: FileText, label: "Results" }
  ];

  return (
    <div className="flex items-center justify-center gap-2 mb-8">
      {steps.map((step, idx) => {
        const Icon = step.icon;
        const isActive = idx === currentStep;
        const isCompleted = idx < currentStep;
        
        return (
          <div key={idx} className="flex items-center">
            <motion.div
              initial={false}
              animate={{
                scale: isActive ? 1.1 : 1,
                backgroundColor: isCompleted ? "#0d9488" : isActive ? "#14b8a6" : "#e2e8f0",
              }}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
                isActive ? "ring-4 ring-teal-200 shadow-lg shadow-teal-500/30" : ""
              }`}
            >
              <Icon className={`w-5 h-5 ${isCompleted || isActive ? "text-white" : "text-slate-500"}`} />
            </motion.div>
            <span className={`ml-2 text-sm font-medium hidden sm:block ${
              isActive ? "text-teal-700" : isCompleted ? "text-teal-600" : "text-slate-400"
            }`}>
              {step.label}
            </span>
            {idx < steps.length - 1 && (
              <div className={`w-8 h-0.5 mx-2 rounded-full ${
                isCompleted ? "bg-teal-500" : "bg-slate-200"
              }`} />
            )}
          </div>
        );
      })}
    </div>
  );
};

// Radial Progress Component
const RadialProgress = ({ percentage, color }: { percentage: number; color: string }) => {
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative w-32 h-32">
      <svg className="transform -rotate-90 w-full h-full">
        <circle
          cx="64"
          cy="64"
          r="45"
          stroke="currentColor"
          strokeWidth="8"
          fill="transparent"
          className="text-slate-200"
        />
        <motion.circle
          cx="64"
          cy="64"
          r="45"
          stroke={color}
          strokeWidth="8"
          fill="transparent"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-2xl font-bold text-slate-800">{percentage.toFixed(0)}%</span>
      </div>
    </div>
  );
};

// Image Compare Slider Component
const ImageCompare = ({ original, analyzed }: { original: string; analyzed: string }) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  return (
    <div 
      ref={containerRef}
      className="relative w-full aspect-[4/3] rounded-2xl overflow-hidden cursor-ew-resize select-none shadow-2xl"
      onMouseMove={(e) => e.buttons === 1 && handleMove(e.clientX)}
      onTouchMove={(e) => handleMove(e.touches[0].clientX)}
    >
      {/* Analyzed Image (Full) */}
      <img src={analyzed} alt="Analyzed" className="absolute inset-0 w-full h-full object-cover" />
      
      {/* Original Image (Clipped) */}
      <div 
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
      >
        <img src={original} alt="Original" className="absolute inset-0 w-full h-full object-cover" />
      </div>

      {/* Slider Line */}
      <div 
        className="absolute top-0 bottom-0 w-1 bg-white shadow-lg"
        style={{ left: `${sliderPosition}%` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 bg-white rounded-full shadow-xl flex items-center justify-center">
          <div className="flex gap-1">
            <ChevronRight className="w-4 h-4 text-slate-600 rotate-180" />
            <ChevronRight className="w-4 h-4 text-slate-600" />
          </div>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm">
        Original
      </div>
      <div className="absolute top-4 right-4 bg-teal-600/80 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm">
        AI Analysis
      </div>
    </div>
  );
};

// Glass Card Component
const GlassCard = ({ children, className = "" }: { children: React.ReactNode; className?: string }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`bg-white/80 backdrop-blur-xl border border-white/20 shadow-xl ${className}`}
  >
    {children}
  </motion.div>
);

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("upload");
  const [showCamera, setShowCamera] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Simulate progress during analysis
  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 300);
      return () => clearInterval(interval);
    } else {
      setAnalysisProgress(0);
    }
  }, [loading]);

  // Camera functions
  const startCamera = useCallback(async () => {
    try {
      // Simple camera access - try rear first, then front
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
      } catch (rearErr) {
        console.log("Rear camera not available, trying front camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play();
        setShowCamera(true);
      }
    } catch (err) {
      console.error("Camera error:", err);
      setError("Camera access denied or not available. Please use file upload instead.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setShowCamera(false);
  }, []);

  const capturePhoto = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const capturedFile = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
            setFile(capturedFile);
            setPreview(canvas.toDataURL('image/jpeg'));
            setPrediction(null);
            setError(null);
            stopCamera();
          }
        }, 'image/jpeg', 0.95);
      }
    }
  }, [stopCamera]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPrediction(null);
      setError(null);
      setCurrentStep(0); // Reset to upload step
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
        setCurrentStep(1); // Move to scan step
        toast.success("Image uploaded successfully!");
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setCurrentStep(2); // Analyzing step
    toast.info("AI analysis in progress...", { duration: 2000 });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const result = await response.json();
      setPrediction(result);
      setCurrentStep(3); // Results step
      setActiveTab("results");
      
      // Toast based on risk level
      if (result.risk_level === "HIGH") {
        toast.error("High risk detected. Please consult a dermatologist.", { 
          duration: 5000,
          icon: <AlertTriangle className="w-5 h-5" />
        });
      } else if (result.risk_level === "LOW") {
        toast.success("Analysis complete. Low risk detected.", { 
          duration: 3000,
          icon: <CheckCircle2 className="w-5 h-5" />
        });
      } else {
        toast.info("Analysis complete. Medium risk - monitor recommended.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      toast.error("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getRiskStyles = (level: string) => {
    switch (level) {
      case "HIGH": 
        return { 
          bg: 'bg-red-50', 
          border: 'border-red-200', 
          text: 'text-red-700',
          badge: 'bg-red-100 text-red-700 border-red-200'
        };
      case "MEDIUM": 
        return { 
          bg: 'bg-amber-50', 
          border: 'border-amber-200', 
          text: 'text-amber-700',
          badge: 'bg-amber-100 text-amber-700 border-amber-200'
        };
      case "LOW": 
        return { 
          bg: 'bg-emerald-50', 
          border: 'border-emerald-200', 
          text: 'text-emerald-700',
          badge: 'bg-emerald-100 text-emerald-700 border-emerald-200'
        };
      default: 
        return { 
          bg: 'bg-slate-50', 
          border: 'border-slate-200', 
          text: 'text-slate-700',
          badge: 'bg-slate-100 text-slate-700 border-slate-200'
        };
    }
  };

  const getConfidenceStyles = (confidence: number) => {
    if (confidence > 0.7) return { color: 'text-emerald-600', bg: 'bg-emerald-500' };
    if (confidence > 0.4) return { color: 'text-amber-600', bg: 'bg-amber-500' };
    return { color: 'text-red-600', bg: 'bg-red-500' };
  };

  const getMedicalRecommendation = (predictedClass: string, confidence: number) => {
    const info = lesionInfo[predictedClass as keyof typeof lesionInfo];
    if (!info) return { text: "Consult a dermatologist for proper evaluation", icon: Stethoscope, urgency: 'normal', color: '#64748b' };
    
    if (info.risk === "HIGH" && confidence > 0.6) {
      return { 
        text: "Immediate dermatologist consultation required. This appears to be a high-risk lesion.", 
        icon: AlertTriangle, 
        urgency: 'urgent',
        color: '#dc2626'
      };
    } else if (info.risk === "HIGH") {
      return { 
        text: "Seek medical evaluation soon. Uncertain high-risk prediction requires professional assessment.", 
        icon: Clock, 
        urgency: 'high',
        color: '#d97706'
      };
    } else if (info.risk === "MEDIUM" && confidence > 0.7) {
      return { 
        text: "Schedule dermatologist appointment for professional evaluation and monitoring.", 
        icon: Stethoscope, 
        urgency: 'medium',
        color: '#0284c7'
      };
    } else {
      return { 
        text: "Continue regular skin examinations. Routine monitoring recommended.", 
        icon: CheckCircle2, 
        urgency: 'low',
        color: '#059669'
      };
    }
  };

  const handleDownloadReport = () => {
    if (!prediction) return;
    
    const report = {
      timestamp: new Date().toISOString(),
      prediction: prediction.predicted_class_full,
      confidence: (prediction.confidence * 100).toFixed(1) + "%",
      risk_level: prediction.risk_level,
      all_probabilities: prediction.all_probabilities,
      recommendation: getMedicalRecommendation(prediction.predicted_class, prediction.confidence),
      disclaimer: "This analysis is for educational purposes only. Always consult qualified healthcare professionals for medical advice."
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'skin-lesion-report.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success("Report downloaded successfully!");
  };

  const handleShare = async () => {
    if (!prediction) return;
    
    const shareData = {
      title: 'SkinXAI Analysis Results',
      text: `My skin lesion analysis: ${prediction.predicted_class_full} (${prediction.risk_level} Risk)`,
      url: window.location.href
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
        toast.success("Shared successfully!");
      } else {
        await navigator.clipboard.writeText(window.location.href);
        toast.success("Link copied to clipboard!");
      }
    } catch {
      // User cancelled share
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-teal-50/30 to-cyan-50/30 relative overflow-hidden">
      <Toaster position="top-center" richColors />
      <ParticleBackground />
      
      {/* Elegant Glass Header */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-white/80 backdrop-blur-xl border-b border-white/50 sticky top-0 z-50 shadow-sm"
      >
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.div 
              className="flex items-center gap-3"
              whileHover={{ scale: 1.02 }}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center shadow-lg shadow-teal-500/30">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-slate-900">SkinXAI</h1>
                <p className="text-xs text-slate-500">AI-Powered Skin Analysis</p>
              </div>
            </motion.div>
            <div className="flex items-center gap-4">
              <Badge variant="outline" className="hidden sm:flex items-center gap-1 border-teal-200 bg-teal-50 text-teal-700">
                <Activity className="w-3 h-3" />
                88.28% Accuracy
              </Badge>
              <Badge variant="outline" className="hidden sm:flex items-center gap-1 border-cyan-200 bg-cyan-50 text-cyan-700">
                <Shield className="w-3 h-3" />
                HIPAA Compliant
              </Badge>
            </div>
          </div>
        </div>
      </motion.header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Step Indicator */}
        <StepIndicator currentStep={currentStep} />
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full max-w-md mx-auto mb-8 bg-white/80 backdrop-blur p-1 rounded-xl border border-slate-200 shadow-sm">
            <TabsTrigger 
              value="upload" 
              className="rounded-lg data-[state=active]:bg-teal-600 data-[state=active]:text-white"
            >
              <ScanLine className="w-4 h-4 mr-2" />
              Scan
            </TabsTrigger>
            <TabsTrigger 
              value="results" 
              disabled={!prediction}
              className="rounded-lg data-[state=active]:bg-teal-600 data-[state=active]:text-white"
            >
              <FileText className="w-4 h-4 mr-2" />
              Results
            </TabsTrigger>
            <TabsTrigger 
              value="info" 
              className="rounded-lg data-[state=active]:bg-teal-600 data-[state=active]:text-white"
            >
              <Info className="w-4 h-4 mr-2" />
              About
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            {/* Main Upload Card */}
            <Card className="border-0 shadow-xl bg-white overflow-hidden">
              <CardHeader className="bg-gradient-to-r from-teal-600 via-cyan-600 to-teal-600 text-white p-8">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur flex items-center justify-center">
                    <ScanLine className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl sm:text-3xl font-bold text-white">Analyze Skin Lesion</CardTitle>
                    <CardDescription className="text-teal-100 text-base mt-1">
                      Take a photo or upload an image for AI-powered analysis
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="p-8">
                {!showCamera ? (
                  <div className="space-y-6">
                    {/* Preview Area */}
                    {preview ? (
                      <div className="relative">
                        <div className="relative rounded-2xl overflow-hidden shadow-lg border border-slate-200">
                          <img
                            src={preview}
                            alt="Preview"
                            className="w-full max-h-96 object-contain bg-slate-100"
                          />
                          <button
                            onClick={clearImage}
                            className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white/90 hover:bg-white shadow-lg flex items-center justify-center transition-all"
                          >
                            <X className="w-5 h-5 text-slate-700" />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="grid sm:grid-cols-2 gap-4">
                        {/* Camera Option */}
                        <button
                          onClick={startCamera}
                          className="group relative rounded-2xl border-2 border-dashed border-slate-300 hover:border-teal-500 bg-slate-50 hover:bg-teal-50/50 transition-all p-8 text-center"
                        >
                          <div className="w-16 h-16 rounded-2xl bg-teal-100 group-hover:bg-teal-200 flex items-center justify-center mx-auto mb-4 transition-colors">
                            <Camera className="w-8 h-8 text-teal-600" />
                          </div>
                          <h3 className="text-lg font-semibold text-slate-900 mb-2">Take Photo</h3>
                          <p className="text-sm text-slate-500">Use your camera for instant analysis</p>
                        </button>

                        {/* Upload Option */}
                        <label className="group relative rounded-2xl border-2 border-dashed border-slate-300 hover:border-cyan-500 bg-slate-50 hover:bg-cyan-50/50 transition-all p-8 text-center cursor-pointer">
                          <input
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            className="hidden"
                          />
                          <div className="w-16 h-16 rounded-2xl bg-cyan-100 group-hover:bg-cyan-200 flex items-center justify-center mx-auto mb-4 transition-colors">
                            <Upload className="w-8 h-8 text-cyan-600" />
                          </div>
                          <h3 className="text-lg font-semibold text-slate-900 mb-2">Upload Image</h3>
                          <p className="text-sm text-slate-500">JPG, PNG files supported</p>
                        </label>
                      </div>
                    )}

                    {/* Hidden canvas for camera capture */}
                    <canvas ref={canvasRef} className="hidden" />

                    {/* Analyze Button */}
                    {file && (
                      <Button
                        onClick={handleAnalyze}
                        disabled={loading}
                        className="w-full h-14 text-lg bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white shadow-lg shadow-teal-500/25 rounded-xl"
                      >
                        {loading ? (
                          <span className="flex items-center gap-3">
                            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            Analyzing with AI...
                          </span>
                        ) : (
                          <span className="flex items-center gap-3">
                            <ScanLine className="w-5 h-5" />
                            Start Analysis
                            <ChevronRight className="w-5 h-5" />
                          </span>
                        )}
                      </Button>
                    )}

                    {error && (
                      <Alert variant="destructive" className="rounded-xl">
                        <AlertTriangle className="w-4 h-4" />
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    )}

                    {/* Instructions */}
                    <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                      <h4 className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                        <Info className="w-4 h-4 text-teal-600" />
                        For best results:
                      </h4>
                      <ul className="space-y-2 text-sm text-slate-600">
                        <li className="flex items-start gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-2 flex-shrink-0" />
                          Ensure good lighting - natural daylight is best
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-2 flex-shrink-0" />
                          Keep the lesion centered and in focus
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-2 flex-shrink-0" />
                          Include some surrounding skin for context
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="w-1.5 h-1.5 rounded-full bg-teal-500 mt-2 flex-shrink-0" />
                          Avoid shadows and reflections
                        </li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  /* Camera View */
                  <div className="space-y-4">
                    <div className="relative rounded-2xl overflow-hidden bg-black min-h-[300px]">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 pointer-events-none">
                        <div className="absolute inset-0 border-2 border-white/30 m-8 rounded-2xl" />
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 border-2 border-white/50 rounded-full" />
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <Button
                        onClick={capturePhoto}
                        className="flex-1 h-14 bg-teal-600 hover:bg-teal-700 text-white rounded-xl"
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        Capture
                      </Button>
                      <Button
                        onClick={stopCamera}
                        variant="outline"
                        className="h-14 px-6 rounded-xl"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Stats Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: 'Accuracy', value: '88.28%', color: 'text-teal-600', bg: 'bg-teal-50' },
                { label: 'Sensitivity', value: '85.2%', color: 'text-cyan-600', bg: 'bg-cyan-50' },
                { label: 'Specificity', value: '91.7%', color: 'text-sky-600', bg: 'bg-sky-50' },
                { label: 'AUC Score', value: '93.4%', color: 'text-indigo-600', bg: 'bg-indigo-50' },
              ].map((stat) => (
                <div key={stat.label} className={`${stat.bg} rounded-xl p-4 text-center border border-slate-200`}>
                  <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
                  <div className="text-xs text-slate-600 mt-1">{stat.label}</div>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <AnimatePresence>
              {prediction && (
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -30 }}
                  transition={{ duration: 0.5 }}
                >
                {/* Primary Result Card */}
                <Card className="border-0 shadow-2xl overflow-hidden bg-white/90 backdrop-blur-xl">
                  <div className={`p-8 ${getRiskStyles(prediction.risk_level).bg} border-b ${getRiskStyles(prediction.risk_level).border}`}>
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-3 mb-4">
                          <motion.div 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ delay: 0.2, type: "spring" }}
                            className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${getRiskStyles(prediction.risk_level).badge}`}
                          >
                            {prediction.risk_level} Risk
                          </motion.div>
                          {prediction.confidence > 0.8 && (
                            <Badge variant="outline" className="bg-white/50 border-emerald-200 text-emerald-700">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              High Confidence
                            </Badge>
                          )}
                        </div>
                        <h2 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-3">
                          {prediction.predicted_class_full}
                        </h2>
                      </div>
                      
                      {/* Radial Progress */}
                      <motion.div
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ delay: 0.3, type: "spring", stiffness: 100 }}
                        className="flex-shrink-0"
                      >
                        <RadialProgress 
                          percentage={prediction.confidence * 100} 
                          color={prediction.risk_level === 'HIGH' ? '#dc2626' : prediction.risk_level === 'MEDIUM' ? '#d97706' : '#059669'} 
                        />
                      </motion.div>
                    </div>
                  </div>
                  
                  <CardContent className="p-8">
                    {/* Medical Recommendation */}
                    {(() => {
                      const rec = getMedicalRecommendation(prediction.predicted_class, prediction.confidence);
                      const Icon = rec.icon;
                      return (
                        <div className={`rounded-xl p-5 border-2 mb-8 ${
                          rec.urgency === 'urgent' ? 'bg-red-50 border-red-200' :
                          rec.urgency === 'high' ? 'bg-amber-50 border-amber-200' :
                          rec.urgency === 'medium' ? 'bg-blue-50 border-blue-200' :
                          'bg-emerald-50 border-emerald-200'
                        }`}>
                          <div className="flex items-start gap-4">
                            <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
                              rec.urgency === 'urgent' ? 'bg-red-100' :
                              rec.urgency === 'high' ? 'bg-amber-100' :
                              rec.urgency === 'medium' ? 'bg-blue-100' :
                              'bg-emerald-100'
                            }`}>
                              <Icon className={`w-6 h-6 ${
                                rec.urgency === 'urgent' ? 'text-red-600' :
                                rec.urgency === 'high' ? 'text-amber-600' :
                                rec.urgency === 'medium' ? 'text-blue-600' :
                                'text-emerald-600'
                              }`} />
                            </div>
                            <div>
                              <h4 className={`font-semibold mb-1 ${
                                rec.urgency === 'urgent' ? 'text-red-900' :
                                rec.urgency === 'high' ? 'text-amber-900' :
                                rec.urgency === 'medium' ? 'text-blue-900' :
                                'text-emerald-900'
                              }`}>
                                Medical Recommendation
                              </h4>
                              <p className={`text-sm ${
                                rec.urgency === 'urgent' ? 'text-red-700' :
                                rec.urgency === 'high' ? 'text-amber-700' :
                                rec.urgency === 'medium' ? 'text-blue-700' :
                                'text-emerald-700'
                              }`}>
                                {rec.text}
                              </p>
                            </div>
                          </div>
                        </div>
                      );
                    })()}

                    {/* Images */}
                    <div className="grid md:grid-cols-2 gap-6 mb-8">
                      <div className="space-y-3">
                        <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                          <ScanLine className="w-4 h-4 text-teal-600" />
                          Original Image
                        </h3>
                        <div className="rounded-xl overflow-hidden border border-slate-200 shadow-md">
                          {preview && (
                            <img
                              src={preview}
                              alt="Original"
                              className="w-full object-cover"
                            />
                          )}
                        </div>
                      </div>
                      <div className="space-y-3">
                        <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                          <Activity className="w-4 h-4 text-cyan-600" />
                          AI Analysis (Grad-CAM)
                        </h3>
                        <div className="rounded-xl overflow-hidden border border-slate-200 shadow-md">
                          <img
                            src={`data:image/png;base64,${prediction.grad_cam_image}`}
                            alt="Grad-CAM"
                            className="w-full object-cover"
                          />
                        </div>
                        <p className="text-xs text-slate-500">
                          Red areas indicate regions the AI focused on for its decision
                        </p>
                      </div>
                    </div>

                    {/* All Probabilities */}
                    <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                      <h3 className="font-semibold text-slate-900 mb-5 flex items-center gap-2">
                        <Activity className="w-4 h-4 text-teal-600" />
                        Complete Analysis
                      </h3>
                      <div className="space-y-4">
                        {prediction.all_probabilities.map((prob, idx) => (
                          <div key={idx} className="space-y-2">
                            <div className="flex justify-between items-center text-sm">
                              <span className="font-medium text-slate-700">
                                {prob.class_name}
                              </span>
                              <span className="font-semibold text-slate-900">
                                {(prob.probability * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-500 ${
                                  idx === 0 ? getConfidenceStyles(prob.probability).bg : 'bg-slate-400'
                                }`}
                                style={{ width: `${prob.probability * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button
                    onClick={handleDownloadReport}
                    className="flex-1 h-12 bg-teal-600 hover:bg-teal-700 text-white rounded-xl"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                  <Button
                    onClick={() => {
                      clearImage();
                      setActiveTab("upload");
                    }}
                    variant="outline"
                    className="flex-1 h-12 rounded-xl"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Analyze Another
                  </Button>
                </div>
              </motion.div>
              )}
            </AnimatePresence>
          </TabsContent>

          <TabsContent value="info" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* About Card */}
              <Card className="border-0 shadow-lg">
                <CardHeader className="bg-gradient-to-r from-teal-600 to-cyan-600 text-white">
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    About SkinXAI
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6 space-y-4">
                  <p className="text-slate-600 leading-relaxed">
                    SkinXAI uses an advanced EfficientNet-B7 deep learning model trained on over 10,000 
                    dermatoscopic images from the HAM10000 dataset. The system classifies skin lesions 
                    into 7 categories with explainable AI visualizations.
                  </p>
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                    <h4 className="font-semibold text-slate-900 mb-2">Key Features</h4>
                    <ul className="space-y-2 text-sm text-slate-600">
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-teal-600 mt-0.5" />
                        Real-time AI analysis with Grad-CAM visualization
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-teal-600 mt-0.5" />
                        Medical-grade risk assessment
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-teal-600 mt-0.5" />
                        Downloadable clinical reports
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-teal-600 mt-0.5" />
                        Privacy-first: no data storage
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              {/* Lesion Types */}
              <Card className="border-0 shadow-lg">
                <CardHeader className="bg-slate-100 border-b border-slate-200">
                  <CardTitle className="text-slate-900 flex items-center gap-2">
                    <Stethoscope className="w-5 h-5" />
                    Lesion Classifications
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="divide-y divide-slate-100">
                    {Object.entries(lesionInfo).map(([key, info]) => (
                      <div key={key} className="p-4 hover:bg-slate-50 transition-colors">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <h4 className="font-semibold text-slate-900">{info.name}</h4>
                            <p className="text-xs text-slate-500 mt-1">{info.description}</p>
                          </div>
                          <Badge 
                            variant="outline" 
                            className={`flex-shrink-0 ${
                              info.risk === 'HIGH' ? 'border-red-200 text-red-700 bg-red-50' :
                              info.risk === 'MEDIUM' ? 'border-amber-200 text-amber-700 bg-amber-50' :
                              'border-emerald-200 text-emerald-700 bg-emerald-50'
                            }`}
                          >
                            {info.risk}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Disclaimer */}
            <Alert className="border-amber-200 bg-amber-50 rounded-xl">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              <AlertDescription className="text-amber-800">
                <strong className="block mb-1">Medical Disclaimer</strong>
                This tool is for educational and research purposes only. It is NOT a substitute for 
                professional medical diagnosis. Always consult qualified healthcare professionals 
                for medical advice. Never use this tool to make medical decisions.
              </AlertDescription>
            </Alert>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-teal-600" />
              <span className="font-semibold text-slate-900">SkinXAI</span>
              <span className="text-slate-400">|</span>
              <span className="text-sm text-slate-500">AI-Powered Skin Analysis</span>
            </div>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <Shield className="w-4 h-4" />
                Privacy Protected
              </span>
              <span>•</span>
              <span>For Educational Use Only</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
