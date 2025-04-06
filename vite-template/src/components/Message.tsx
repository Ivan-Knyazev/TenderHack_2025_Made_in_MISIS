import { Card, CardBody } from "@heroui/card";
import { useState } from "react";
import { Rate } from "antd";
// import {Spinner} from "@heroui/spinner";
import { Spinner } from "@heroui/spinner";

import getAlert from "../helpers/getAlert";
import { BACKEND_URL } from "../consts";

type MessagePropsType = {
  text: string;
  source: string[];
  type: "bot" | "user";
  position: string;
  id: "string";
  isFetching: { isFetching: boolean; id: string };
};

const Message = ({
  text,
  source,
  type,
  position,
  id,
  isFetching,
  ...props
}: MessagePropsType) => {
  const [isEstimated, setIsEstimated] = useState<boolean>(
    type === "bot" ? true : false,
  );
  const [rate, setRate] = useState<number>(-1);
  const [isAnswered, setIsAnswered] = useState<boolean>(false);

  console.log("id:", id, 'mes_id:', isFetching.id, isFetching.isFetching);

  const sources = source?.map(item => {
    return <span className="color-blue-500">{item}</span>;
  });

  console.log('source:', source);

  const handleRate = (e: number) => {
    console.log("edit");
    setRate(e);
    fetch(`${BACKEND_URL}/api/v1/query/rate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ rate: e, query_id: id }),
    })
      .then((req) => {
        if (!req.ok) {
          throw new Error("Ошибка при соединении");
        }
      })
      .then((res) => {
        console.log(res);
        setIsEstimated(false);
        setIsAnswered(true);
        setTimeout(() => {
          setIsAnswered(false);
        }, 5000);
      })
      .catch((e) => {
        console.log("yes");
        setIsEstimated(false);
        setIsAnswered(true);
        setTimeout(() => {
          setIsAnswered(false);
        }, 5000);
        getAlert("Ошибка при отправки оценки", e.message, "alerts");
      });
  };

  return (
    <>
      {position === "end" && (
        <div className="w-[100%] flex flex-row justify-end bg-slate-500 text-right">
          <Card className="flex w-[520px]">
            <CardBody>
              {text}
              <div className="m-[12px]">
                {isEstimated && (
                  <Rate
                    allowClear={false}
                    defaultValue={-1}
                    onChange={(e) => {
                      handleRate(e);
                    }}
                  />
                )}
                {isAnswered && (
                  <span className="font-[12px]">Спасибо за оценку!</span>
                )}
                {isFetching?.isFetching && isFetching?.id === id && <Spinner size="sm"/>}
              </div>
            </CardBody>
          </Card>
        </div>
      )}
      {position === "start" && (
        <div className="w-[100%] flex flex-row justify-start text-left">
          <Card className="flex w-[560px]">
            <CardBody>
              {text}
              <div className="text-cyan-600">{source && sources}</div>
              <div className="m-[12px]">
                {isEstimated && (
                  <Rate
                    allowClear={false}
                    defaultValue={-1}
                    onChange={(e) => {
                      handleRate(e);
                    }}
                  />
                )}
                {isAnswered && (
                  <span className="font-[12px]">Спасибо за оценку!</span>
                )}
                {/* {isFetching?.isFetching && <Spinner size="sm"/>} */}
              </div>
            </CardBody>
          </Card>
        </div>
      )}
    </>
  );
};

export default Message;
