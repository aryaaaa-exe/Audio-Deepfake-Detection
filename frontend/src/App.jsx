import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import UploadBox from "./components/UploadBox";
import Background from "./components/Background";
import ModelExplanation from "./components/ModelExplanation";
//import ProcessingPipeline from "./components/ProcessingPipeline";
import WhyPS3DT from "./components/WhyPS3DT";
import Footer from "./components/Footer";
import FadeIn from "./animations/FadeIn";

import "./styles/Home.css";

function App() {
  return (
    <div className="app">
    <>
      <Background/>
      
      <Navbar />

      <Hero />

      <UploadBox />

      <ModelExplanation />

      <WhyPS3DT />

      <Footer />

    </>
    </div>
  );
}

export default App;