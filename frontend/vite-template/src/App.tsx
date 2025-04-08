import { Route, Routes } from "react-router-dom";
import ChatbotPage from "./pages/ChatbotPage";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import RegistrationPage from "./pages/RegistrationPage";


function App() {
  return (
    <Routes>
          <Route path="/chat" element={<ChatbotPage/>}/>
          <Route path="/registration" element={<RegistrationPage/>}/>
          <Route path="/login" element={<LoginPage/>}/>
          <Route path="/admin-panel" element={<DashboardPage/>}/>
        </Routes>
  );
}

export default App;
