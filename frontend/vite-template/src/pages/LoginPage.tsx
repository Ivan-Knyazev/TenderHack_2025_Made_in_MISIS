import { Button } from "@heroui/button";
import {Input} from '@heroui/input';
import { useState } from "react";
import { BACKEND_URL } from "../consts";

const LoginPage = () => {
    const [form, setForm] = useState({login: '', password: ''});
    const [isLogged, setIsLogged] = useState<boolean>(false);

    const handleClick = (e) => {
         fetch(`${BACKEND_URL}/login`, {
                method: 'POST',
                headers: {
                    "Content-Type": "application/json",
                  },
                body: JSON.stringify({text, user_id})
            }).then(res => res.json())
            .then(ans => {setLogged(true)})
            .catch((e) => getAlert(e.message))
    }

    return(
    <div className="flex flex-col gap-[20px]">
        <Input label="Email" 
        onValueChange={(e) => setForm({...form, login: e})}
        placeholder="Enter your email" type="email" />
        <Input label="Email" 
        onValueChange={(e) => setForm({...form, password: e})}
        placeholder="Enter your email" type="password"/>

        <Button onPress={(e) => {handleClick(e)}}></Button>

    </div>)

}

export default LoginPage;