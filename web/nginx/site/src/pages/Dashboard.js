import React from 'react';
import TopMenu from "../TopMenu";
import {Container, Divider, Grid, Header, Search, Segment} from "semantic-ui-react";
import Footer from "../Footer";
import {BASE_URL, ensureSignedIn, fetchUsers, getJWT, hasJWT, refreshJWT} from "../Util";
import _ from 'lodash'
import logo from "../icon.svg";
import {SemanticToastContainer} from 'react-semantic-toasts';
import AgentGrid from "../components/AgentGrid";
import CreateAgentModal from "../components/CreateAgentModal";
import RESTPagination from "../components/RESTPagination";


export default class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      agents: [],
      users: {},
      agentCount: 0,
    };
    this.pageSize = 5;
  }

  handleResultSelect = (e, {result}) => this.setState({value: result.title});

  handleSearchChange = (e, {value}) => {
  };

  componentDidMount = () => {
    ensureSignedIn();
    this.fetchAgents(1);
  };

  fetchAgents = (page) => {
    if (!hasJWT()) {
      return;
    }

    fetch(BASE_URL + '/api/agents/?page_size=' + this.pageSize + '&page=' + page, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'JWT ' + getJWT(),
      },
    }).then(response => {
      if (response.status === 401) {
        refreshJWT();
        return;
      }

      if (response.status !== 200) {
        response.text().then(body => {
          console.error("Unable to fetch agents: ", body);
        });
        return;
      }

      response.json().then(body => {
        this.setState({
          agents: body.results,
          agentCount: body.count,
        });
        let userURIs = new Set();
        this.state.agents.forEach(agent => {
          userURIs.add(agent['author']);
        });
        fetchUsers(userURIs, users => {
          this.setState({users: users})
        });
      });
    });
  };

  getPagination = () => {
    return (
      <RESTPagination
        itemCount={this.state.agentCount}
        pageSize={this.pageSize}
        onPageChange={(page) => this.fetchAgents(page)}
        hideIfOnePage
      />
    );
  };

  render = () => (
    <div className='SitePage'>
      <TopMenu/>
      <SemanticToastContainer position='bottom-right'/>
      <Container className='SiteContents' style={{marginTop: '80px', marginBottom: '32px'}}>
        <Header as='h2' color='teal' textAlign='center'>
          <img src={logo} alt='logo' className='image'/>
          {' '} Agents
        </Header>
        <Divider/>
        <Segment basic textAlign='center'>
          <Grid columns={2} relaxed='very'>
            <Grid.Column>
              <Search
                placeholder='Search agents'
                loading={this.state.isLoading}
                onResultSelect={this.handleResultSelect}
                onSearchChange={_.debounce(this.handleSearchChange, 500, {
                  leading: true,
                })}
                results={this.state.results}
                value={this.state.value}
              />
            </Grid.Column>
            <Grid.Column>
              <CreateAgentModal onCreate={() => this.fetchAgents(1)}/>
            </Grid.Column>
          </Grid>
          <Divider vertical>Or</Divider>
        </Segment>
        {this.getPagination()}
        <AgentGrid agents={this.state.agents} users={this.state.users}/>
        {this.getPagination()}
      </Container>
      < Footer/>
    </div>
  );
}
