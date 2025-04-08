import { VictoryBar, VictoryChart, VictoryLegend, VictoryPie, VictoryTheme } from "victory";
import { mocksData, mockVictoryPie } from "../mocks";
import { useEffect, useState } from "react";
import { Tab, Tabs } from "@heroui/tabs";
import TableCustom from "../components/table/TableCustom";
import { BACKEND_URL } from "@/consts";

// Подготовка данных
const data = [
  { x: 1, y: 2 }, { x: 2, y: 2 }, { x: 3, y: 3 }, { x: 4, y: 2 }, { x: 5, y: 2 }
];

const legendData = [
  { name: "Навигация и функциональность портала" },
  { name: "Технические проблемы" },
  { name: "Документы и инструкции" },
  { name: "Законодательство и нормативка" },
  { name: "Обратная связь и жалобы" }
];

// Цветовая схема для графиков
const colorScale = ["tomato", "orange", "gold", "cyan", "green"];

// Компонент Tabs
const TabsContent = () => {
  const [datas, setData] = useState({});

  useEffect(() => {
    fetch(`${BACKEND_URL}/api/v1/query/analitycs`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((ans) => {
        setData(ans);
      })
      .catch((e) => {
        console.error("Ошибка при получении данных:", e);
      });
  }, []);

  const tabs = [
    <div className="grid grid-cols-2 grid-rows-1" key={0}>
      <div>
        <VictoryChart
          domainPadding={{ x: 20 }}
          theme={VictoryTheme.material}
          // padding={{ top: 20, bottom: 60, left: 50, right: 50 }}
          height={200}
          width={300}
        >
          <VictoryBar data={datas?.bars ?? data} colorScale={colorScale} />
        </VictoryChart>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0px' }}>
        <VictoryLegend
          height={20}
          orientation="horizontal"
          style={{
            labels: {
              fontSize: 8,
              fontFamily: 'Helvetica, Arial, sans-serif',
            },
            border: { stroke: "grey" },
          }}
          data={legendData.map((s, index) => ({
            name: s.name,
            symbol: {
              fill: colorScale[index],
              type: "circle",
            },
          }))}
        />
        <VictoryPie
          data={datas?.percents ?? data}
          theme={VictoryTheme.material}
          colorScale={colorScale}
          labels={() => null}
          height={300}
          width={400}
        />
      </div>
    </div>,
    <span key={1}>
      Таблица
      <TableCustom />
    </span>
  ];

  return (
    <div className="flex w-full flex-col">
      <Tabs
        aria-label="Options"
        classNames={{
          tabList: "gap-6 w-full relative rounded-none p-0 border-b border-divider",
          cursor: "w-full bg-[#22d3ee]",
          tab: "max-w-fit px-0 h-12",
          tabContent: "group-data-[selected=true]:text-[#06b6d4]",
        }}
        color="primary"
        variant="underlined"
      >
        <Tab
          key="dashboard"
          title={
            <div className="flex items-center space-x-2">
              <span>Дашборд аналитики</span>
            </div>
          }
        >
          {tabs[0]}
        </Tab>
        <Tab
          key="history"
          title={
            <div className="flex items-center space-x-2">
              <span>История запросов</span>
            </div>
          }
        >
          {tabs[1]}
        </Tab>
      </Tabs>
    </div>
  );
}

export default TabsContent;
